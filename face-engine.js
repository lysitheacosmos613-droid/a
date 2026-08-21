/*
 * face-engine.js — 顔検出・照合の共通処理
 *
 * face-auth.html（顔認証システム）と kintai.html（勤怠システム）で共有する。
 * face-api.js（vendor/face-api/face-api.js）を先に読み込んでおくこと。
 *
 * 顔検出 → 68点ランドマークで位置合わせ → 128次元特徴量 までを担当し、
 * UI・保存・画面表示には関与しない。
 */
(function (global) {
  "use strict";

  // TinyFaceDetector は inputSize によって得意な顔の大きさが変わる。
  // 小さい値ほど「大きく写った顔」、大きい値ほど「小さく写った顔」に強いので、
  // 自動モードでは直近に成功したサイズを優先しつつ候補を巡回させる。
  var SIZE_LADDER = [224, 320, 416, 160, 512];
  var IMAGE_SIZES = [224, 320, 416, 512, 160];

  var DEFAULT_LOCAL_URI = "vendor/face-api/model";
  var DEFAULT_CDN_URI = "https://cdn.jsdelivr.net/npm/@vladmandic/face-api@1.7.15/model";

  function loadFrom(uri) {
    return Promise.all([
      faceapi.nets.tinyFaceDetector.loadFromUri(uri),
      faceapi.nets.faceLandmark68Net.loadFromUri(uri),
      faceapi.nets.faceRecognitionNet.loadFromUri(uri)
    ]);
  }

  // ローカル同梱のモデルを優先し、失敗したら CDN にフォールバックする。
  // onProgress(step) は "local" / "cdn" を受け取る。
  function loadModels(opts) {
    opts = opts || {};
    var local = opts.localUri || DEFAULT_LOCAL_URI;
    var cdn = opts.cdnUri || DEFAULT_CDN_URI;
    var onProgress = opts.onProgress || function () {};
    onProgress("local");
    return loadFrom(local)
      .catch(function (e) {
        console.warn("ローカルのモデル読み込みに失敗、CDNを試します", e);
        onProgress("cdn");
        return loadFrom(cdn);
      })
      .then(function () {
        var backend = "—";
        try { backend = faceapi.tf.getBackend() || "—"; } catch (e) {}
        return backend;
      });
  }

  function tinyOpts(size, scoreThreshold) {
    return new faceapi.TinyFaceDetectorOptions({
      inputSize: Number(size),
      scoreThreshold: Number(scoreThreshold)
    });
  }

  /*
   * 映像用の検出器。getConfig() は
   *   { inputSize: "auto" | 224 | 320 | 416 | 512, scoreThreshold: 0.5 }
   * を返す関数。detect(video) が検出結果（なければ null）を返し、
   * 成功したサイズは detector.activeSize で読める。
   */
  function createDetector(getConfig) {
    var sizeIdx = 1;
    var self = {
      activeSize: null,
      detect: function (input) {
        var config = getConfig();
        var score = Number(config.scoreThreshold);
        var sizes;
        if (config.inputSize !== "auto") {
          sizes = [Number(config.inputSize)];
        } else {
          sizes = [SIZE_LADDER[sizeIdx], SIZE_LADDER[(sizeIdx + 1) % SIZE_LADDER.length]];
        }

        function once(size) {
          return faceapi
            .detectSingleFace(input, tinyOpts(size, score))
            .withFaceLandmarks()
            .withFaceDescriptor();
        }

        return once(sizes[0]).then(function (d) {
          if (d) { self.activeSize = sizes[0]; return d; }
          if (sizes.length < 2) { self.activeSize = null; return null; }
          return once(sizes[1]).then(function (d2) {
            self.activeSize = d2 ? sizes[1] : null;
            sizeIdx = (sizeIdx + 1) % SIZE_LADDER.length;
            return d2;
          });
        });
      }
    };
    return self;
  }

  /*
   * 静止画からの検出。顔の大きさが事前に分からないため、候補サイズを総当たりして
   * 一番よく写るサイズを選び、その中で最も大きい顔を返す。
   * 返り値には faceCount（検出した顔の数）と usedSize（採用した解像度）が付く。
   */
  function detectFromImage(img, scoreThreshold) {
    var score = scoreThreshold == null ? 0.4 : scoreThreshold;
    var best = { size: null, faces: 0, score: 0 };
    var chain = Promise.resolve();
    IMAGE_SIZES.forEach(function (size) {
      chain = chain.then(function () {
        return faceapi.detectAllFaces(img, tinyOpts(size, score)).then(function (dets) {
          if (!dets.length) return;
          var top = dets.reduce(function (m, d) { return d.score > m ? d.score : m; }, 0);
          if (dets.length > best.faces || (dets.length === best.faces && top > best.score)) {
            best = { size: size, faces: dets.length, score: top };
          }
        });
      });
    });
    return chain.then(function () {
      if (!best.size) return null;
      return faceapi
        .detectAllFaces(img, tinyOpts(best.size, score))
        .withFaceLandmarks()
        .withFaceDescriptors()
        .then(function (results) {
          if (!results.length) return null;
          results.sort(function (a, b) {
            return b.detection.box.width * b.detection.box.height -
                   a.detection.box.width * a.detection.box.height;
          });
          var r = results[0];
          r.faceCount = results.length;
          r.usedSize = best.size;
          return r;
        });
    });
  }

  // 128次元特徴量同士のユークリッド距離。小さいほど似ている。
  function distance(a, b) {
    var s = 0;
    for (var i = 0; i < a.length; i++) { var d = a[i] - b[i]; s += d * d; }
    return Math.sqrt(s);
  }

  function meanDescriptor(list) {
    var out = new Array(list[0].length).fill(0);
    list.forEach(function (d) {
      for (var i = 0; i < d.length; i++) out[i] += d[i];
    });
    for (var i = 0; i < out.length; i++) out[i] = out[i] / list.length;
    return out;
  }

  /*
   * 登録者全員と比較して最も近い人を返す。
   * people は [{ descriptors: [[...]], descriptor: [...] }] 形式。
   * 各人の全サンプル中の最小距離を採用する。
   */
  function identify(descriptor, people) {
    var best = { person: null, dist: Infinity };
    for (var i = 0; i < people.length; i++) {
      var p = people[i];
      var list = p.descriptors && p.descriptors.length ? p.descriptors : [p.descriptor];
      var min = Infinity;
      for (var j = 0; j < list.length; j++) {
        var d = distance(descriptor, list[j]);
        if (d < min) min = d;
      }
      if (min < best.dist) best = { person: p, dist: min };
    }
    return best;
  }

  // 開眼度（Eye Aspect Ratio）。まばたきで小さくなる。
  function eyeAspectRatio(pts) {
    var v1 = Math.hypot(pts[1].x - pts[5].x, pts[1].y - pts[5].y);
    var v2 = Math.hypot(pts[2].x - pts[4].x, pts[2].y - pts[4].y);
    var h = Math.hypot(pts[0].x - pts[3].x, pts[0].y - pts[3].y);
    return h === 0 ? 0 : (v1 + v2) / (2 * h);
  }

  function earOf(landmarks) {
    return (eyeAspectRatio(landmarks.getLeftEye()) + eyeAspectRatio(landmarks.getRightEye())) / 2;
  }

  /*
   * まばたき検知（簡易ライブネス）。update(landmarks) を毎フレーム呼び、
   * ok() が直近 windowMs 以内にまばたきがあったかを返す。
   * 注意：写真をかざす程度の単純ななりすましは防げるが、
   * 動画を使った提示攻撃までは防げない。
   */
  function createBlinkDetector(opts) {
    opts = opts || {};
    var closedBelow = opts.closedBelow || 0.19;
    var openAbove = opts.openAbove || 0.25;
    var windowMs = opts.windowMs || 10000;
    var closed = false, lastBlinkAt = 0, blinks = 0;
    return {
      update: function (landmarks) {
        var ear = earOf(landmarks);
        if (!closed && ear < closedBelow) {
          closed = true;
        } else if (closed && ear > openAbove) {
          closed = false;
          blinks++;
          lastBlinkAt = Date.now();
        }
        return ear;
      },
      ok: function () { return Date.now() - lastBlinkAt < windowMs; },
      blinks: function () { return blinks; },
      reset: function () { closed = false; lastBlinkAt = 0; }
    };
  }

  /*
   * 検出した顔の周辺を切り出して JPEG の data URL にする。
   * source は video / img / canvas、box は検出結果の box。
   */
  function cropFace(source, box, size, quality) {
    size = size || 128;
    var sw = source.videoWidth || source.naturalWidth || source.width;
    var sh = source.videoHeight || source.naturalHeight || source.height;
    var m = box.width * 0.28;
    var sx = Math.max(0, Math.min(box.x - m, sw));
    var sy = Math.max(0, Math.min(box.y - m * 1.2, sh));
    var w = Math.max(1, Math.min(box.width + m * 2, sw - sx));
    var h = Math.max(1, Math.min(box.height + m * 2.2, sh - sy));
    var c = document.createElement("canvas");
    c.width = size; c.height = size;
    c.getContext("2d").drawImage(source, sx, sy, w, h, 0, 0, size, size);
    return c.toDataURL("image/jpeg", quality == null ? 0.72 : quality);
  }

  /*
   * 映像の表示領域（object-fit: cover）に検出座標を合わせるための係数。
   * canvas の実サイズを表示サイズに合わせたうえで scale / dx / dy を返す。
   */
  function fitOverlay(video, canvas) {
    var w = video.videoWidth, h = video.videoHeight;
    if (!w || !h) return null;
    var rect = video.getBoundingClientRect();
    if (canvas.width !== Math.round(rect.width) || canvas.height !== Math.round(rect.height)) {
      canvas.width = Math.round(rect.width);
      canvas.height = Math.round(rect.height);
    }
    var scale = Math.max(canvas.width / w, canvas.height / h);
    return { scale: scale, dx: (canvas.width - w * scale) / 2, dy: (canvas.height - h * scale) / 2 };
  }

  global.FaceEngine = {
    SIZE_LADDER: SIZE_LADDER,
    IMAGE_SIZES: IMAGE_SIZES,
    loadModels: loadModels,
    createDetector: createDetector,
    detectFromImage: detectFromImage,
    distance: distance,
    meanDescriptor: meanDescriptor,
    identify: identify,
    eyeAspectRatio: eyeAspectRatio,
    earOf: earOf,
    createBlinkDetector: createBlinkDetector,
    cropFace: cropFace,
    fitOverlay: fitOverlay
  };
})(window);
