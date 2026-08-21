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
      },
      // 顔の位置だけを探す（ランドマークと特徴量は計算しない）
      detectBox: function (input) {
        var config = getConfig();
        var score = Number(config.scoreThreshold);
        var sizes = config.inputSize !== "auto"
          ? [Number(config.inputSize)]
          : [SIZE_LADDER[sizeIdx], SIZE_LADDER[(sizeIdx + 1) % SIZE_LADDER.length]];
        function once(size) { return faceapi.detectSingleFace(input, tinyOpts(size, score)); }
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

  // ------------------------------------------------------------------
  // MediaPipe Face Landmarker（検出 + 478点ランドマーク + 表情スコア）
  // ------------------------------------------------------------------
  /*
   * face-api の TinyFaceDetector + 68点ランドマークの代わりに使う。
   * 顔の大きさによって検出できたりできなかったりする問題がなく、
   * まばたきも表情スコア（blendshapes）として直接得られる。
   *
   * 特徴量の抽出は引き続き face-api の FaceRecognitionNet を使うため、
   * face-api と同じ位置合わせ（dlib方式）を478点から再現している。
   * これにより、登録済みの特徴量をそのまま使い続けられる。
   */

  // MediaPipe Face Mesh の目・口の輪郭インデックス
  var MP_LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246];
  var MP_RIGHT_EYE = [263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466];
  var MP_LIPS = [
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185,
    78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 42, 183
  ];

  function centroid(points, indices, w, h) {
    var sx = 0, sy = 0;
    indices.forEach(function (i) { sx += points[i].x; sy += points[i].y; });
    return { x: (sx / indices.length) * w, y: (sy / indices.length) * h };
  }

  /*
   * face-api（dlib方式）と同じ切り出し矩形を計算する。
   * size = 両目それぞれから口までの距離の平均 / 0.45
   * 左上 = 3点の重心 - (0.5 * size, 0.43 * size)
   * 定数は face-api の実装（relScale=0.45 / relX=0.5 / relY=0.43）に合わせている。
   */
  function alignRect(leftEye, rightEye, mouth, imgW, imgH, scale) {
    var d = function (p) { return Math.hypot(mouth.x - p.x, mouth.y - p.y); };
    var size = Math.floor(((d(leftEye) + d(rightEye)) / 2) / 0.45 * (scale || 1));
    var cx = (leftEye.x + rightEye.x + mouth.x) / 3;
    var cy = (leftEye.y + rightEye.y + mouth.y) / 3;
    var x = Math.floor(Math.max(0, cx - 0.5 * size));
    var y = Math.floor(Math.max(0, cy - 0.43 * size));
    return new faceapi.Rect(x, y, Math.min(size, imgW - x), Math.min(size, imgH - y));
  }

  function landmarkBox(points, w, h) {
    var minX = 1, minY = 1, maxX = 0, maxY = 0;
    points.forEach(function (p) {
      if (p.x < minX) minX = p.x;
      if (p.y < minY) minY = p.y;
      if (p.x > maxX) maxX = p.x;
      if (p.y > maxY) maxY = p.y;
    });
    return new faceapi.Rect(minX * w, minY * h, (maxX - minX) * w, (maxY - minY) * h);
  }

  function blendshapeValue(shapes, name) {
    if (!shapes || !shapes.categories) return 0;
    for (var i = 0; i < shapes.categories.length; i++) {
      if (shapes.categories[i].categoryName === name) return shapes.categories[i].score;
    }
    return 0;
  }

  /*
   * MediaPipe を初期化する。読み込めない環境では null を返し、
   * 呼び出し側は face-api のみの従来経路にそのまま戻れる。
   */
  function createMediaPipe(opts) {
    opts = opts || {};
    var base = opts.baseUri || "vendor/mediapipe";
    var mode = opts.runningMode || "VIDEO";
    var mp = global.__MEDIAPIPE__;
    if (!mp) return Promise.resolve(null);
    return mp.FilesetResolver.forVisionTasks(base + "/wasm")
      .then(function (fileset) {
        return mp.FaceLandmarker.createFromOptions(fileset, {
          baseOptions: { modelAssetPath: base + "/face_landmarker.task", delegate: opts.delegate || "GPU" },
          runningMode: mode,
          numFaces: opts.numFaces || 3,
          outputFaceBlendshapes: true,
          outputFacialTransformationMatrixes: false
        });
      })
      .then(function (landmarker) { return wrapLandmarker(landmarker, mode); })
      .catch(function (e) {
        console.warn("MediaPipe をGPUで初期化できませんでした。CPUで再試行します", e);
        return mp.FilesetResolver.forVisionTasks(base + "/wasm").then(function (fileset) {
          return mp.FaceLandmarker.createFromOptions(fileset, {
            baseOptions: { modelAssetPath: base + "/face_landmarker.task", delegate: "CPU" },
            runningMode: mode, numFaces: opts.numFaces || 3, outputFaceBlendshapes: true
          });
        }).then(function (l) { return wrapLandmarker(l, mode); }).catch(function (e2) {
          console.warn("MediaPipe を初期化できませんでした", e2);
          return null;
        });
      });
  }

  function wrapLandmarker(landmarker, mode) {
    var lastTs = -1;
    function build(res, w, h) {
      if (!res || !res.faceLandmarks || !res.faceLandmarks.length) return null;
      var pick = 0;
      if (res.faceLandmarks.length > 1) {
        var maxArea = -1;
        res.faceLandmarks.forEach(function (p, i) {
          var b = landmarkBox(p, w, h);
          var area = b.width * b.height;
          if (area > maxArea) { maxArea = area; pick = i; }
        });
      }
      var pts = res.faceLandmarks[pick];
      var shapes = res.faceBlendshapes && res.faceBlendshapes[pick];
      var leftEye = centroid(pts, MP_LEFT_EYE, w, h);
      var rightEye = centroid(pts, MP_RIGHT_EYE, w, h);
      var mouth = centroid(pts, MP_LIPS, w, h);
      return {
        box: landmarkBox(pts, w, h),
        alignedRect: alignRect(leftEye, rightEye, mouth, w, h),
        landmarks: pts,
        refPoints: { leftEye: leftEye, rightEye: rightEye, mouth: mouth },
        blink: Math.max(blendshapeValue(shapes, "eyeBlinkLeft"), blendshapeValue(shapes, "eyeBlinkRight")),
        faceCount: res.faceLandmarks.length,
        score: 1
      };
    }
    return {
      raw: landmarker,
      mode: mode,
      // 静止画用（runningMode: "IMAGE" で作った場合のみ使える）
      detectStill: function (img) {
        if (mode !== "IMAGE") return null;
        var w = img.naturalWidth || img.width, h = img.naturalHeight || img.height;
        try { return build(landmarker.detect(img), w, h); }
        catch (e) { console.warn("MediaPipe の検出に失敗", e); return null; }
      },
      /*
       * 映像から顔を1つ検出する。返り値は
       *   { box, alignedRect, landmarks, blink, score }
       * blink は左右の目の閉じ具合（0〜1）。検出できなければ null。
       */
      detect: function (video) {
        var w = video.videoWidth || video.naturalWidth || video.width;
        var h = video.videoHeight || video.naturalHeight || video.height;
        if (!w || !h) return null;
        var ts = performance.now();
        if (ts <= lastTs) ts = lastTs + 1; // MediaPipe は時刻の巻き戻りを許さない
        lastTs = ts;
        var res;
        try { res = landmarker.detectForVideo(video, ts); }
        catch (e) { console.warn("MediaPipe の検出に失敗", e); return null; }
        return build(res, w, h);
      },
      close: function () { try { landmarker.close(); } catch (e) {} }
    };
  }

  /*
   * 位置合わせ済みの矩形から128次元の特徴量を作る。
   * face-api の高レベルAPIと同じ手順（切り出し → FaceRecognitionNet）。
   */
  function describeAligned(input, rect) {
    return faceapi.extractFaces(input, [rect]).then(function (canvases) {
      if (!canvases.length) return null;
      return faceapi.computeFaceDescriptor(canvases[0]).then(function (desc) {
        return Array.from(desc);
      });
    });
  }

  /*
   * まばたき判定（MediaPipe の表情スコア版）。
   * 閉じ具合が closedAbove を超えてから openBelow を下回ったら1回とみなす。
   */
  function createBlendshapeBlinkDetector(opts) {
    opts = opts || {};
    var closedAbove = opts.closedAbove || 0.5;
    var openBelow = opts.openBelow || 0.25;
    var windowMs = opts.windowMs || 10000;
    var closed = false, lastBlinkAt = 0, blinks = 0;
    return {
      update: function (blink) {
        if (!closed && blink > closedAbove) closed = true;
        else if (closed && blink < openBelow) { closed = false; blinks++; lastBlinkAt = Date.now(); }
        return blink;
      },
      ok: function () { return Date.now() - lastBlinkAt < windowMs; },
      blinks: function () { return blinks; },
      reset: function () { closed = false; lastBlinkAt = 0; }
    };
  }

  /*
   * 検出の入口をひとつにまとめたもの。
   *   1) MediaPipe で映像全体から顔を探す（近〜中距離に強い）
   *   2) 見つからなければ face-api の検出器で顔の位置だけ探し、
   *      その周辺を切り出して MediaPipe にかけ直す（遠い顔の救済）
   *   3) MediaPipe が使えない環境では face-api だけの従来経路にそのまま戻る
   *
   * 返り値は face-api の検出結果と同じ形（detection / landmarks / descriptor）に
   * 揃えてあるので、呼び出し側は経路を意識せずに扱える。
   */
  function createReader(opts) {
    opts = opts || {};
    var getConfig = opts.getConfig || function () { return { inputSize: "auto", scoreThreshold: 0.5 }; };
    var legacy = createDetector(getConfig);
    var mp = null, mpImage = null;
    var self = {
      legacy: legacy,
      get usingMediaPipe() { return !!mp; },
      get activeSize() { return legacy.activeSize; },
      lastSource: null,

      init: function () {
        return createMediaPipe(opts.mediapipe || {}).then(function (m) {
          mp = m;
          return !!m;
        });
      },

      readVideo: function (video) {
        if (!mp) {
          return legacy.detect(video).then(function (det) {
            self.lastSource = det ? "face-api" : null;
            return det ? toResult(det, null, "face-api") : null;
          });
        }
        var w = video.videoWidth, h = video.videoHeight;
        var hit = mp.detect(video);
        if (hit) return finish(video, hit, 0, 0, "mediapipe");
        // 遠い顔の救済：face-api で位置を探し、その周りを切り出して再挑戦
        return legacy.detectBox(video).then(function (d) {
          if (!d) { self.lastSource = null; return null; }
          var b = d.box, m = b.width * 0.8;
          var rect = new faceapi.Rect(
            Math.max(0, b.x - m), Math.max(0, b.y - m),
            Math.min(b.width + 2 * m, w), Math.min(b.height + 2 * m, h));
          return faceapi.extractFaces(video, [rect]).then(function (crops) {
            if (!crops.length) return null;
            var hit2 = mp.detect(crops[0]);
            if (!hit2) { self.lastSource = null; return null; }
            return finish(video, hit2, rect.x, rect.y, "mediapipe-crop");
          });
        });
      },

      // 静止画（写真からの登録・照合）用。複数の顔があれば一番大きい顔を返す
      readImage: function (img) {
        var start = mpImage ? Promise.resolve(mpImage) : createMediaPipe(
          Object.assign({}, opts.mediapipe, { runningMode: "IMAGE" })).then(function (m) {
            mpImage = m; return m;
          });
        return start.then(function (m) {
          if (!m) return detectFromImage(img).then(function (d) { return d ? toResult(d, null, "face-api") : null; });
          var hit = m.detectStill ? m.detectStill(img) : null;
          if (!hit) return detectFromImage(img).then(function (d) { return d ? toResult(d, null, "face-api") : null; });
          return finish(img, hit, 0, 0, "mediapipe");
        });
      }
    };

    function finish(input, hit, dx, dy, source) {
      var rect = hit.alignedRect;
      if (dx || dy) rect = new faceapi.Rect(rect.x + dx, rect.y + dy, rect.width, rect.height);
      return describeAligned(input, rect).then(function (desc) {
        if (!desc) { self.lastSource = null; return null; }
        self.lastSource = source;
        var box = hit.box;
        if (dx || dy) box = new faceapi.Rect(box.x + dx, box.y + dy, box.width, box.height);
        return {
          detection: { box: box, score: hit.score },
          landmarks: { positions: hit.landmarks, normalized: true, offset: { x: dx, y: dy } },
          descriptor: desc,
          blink: hit.blink,
          alignedRect: rect,
          faceCount: hit.faceCount,
          source: source
        };
      });
    }

    function toResult(det, blink, source) {
      self.lastSource = source;
      return {
        detection: { box: det.detection.box, score: det.detection.score },
        landmarks: { positions: det.landmarks.positions, normalized: false, offset: { x: 0, y: 0 }, raw: det.landmarks },
        descriptor: Array.from(det.descriptor),
        blink: blink,
        alignedRect: null,
        source: source,
        faceCount: det.faceCount,
        usedSize: det.usedSize
      };
    }

    return self;
  }

  /*
   * まばたき判定（経路の違いを吸収する）。
   * MediaPipe があれば表情スコアを、なければ68点から計算した開眼度(EAR)を使う。
   * update() には createReader の検出結果をそのまま渡す。
   */
  function createLiveness(opts) {
    opts = opts || {};
    var ear = createBlinkDetector(opts);
    var shape = createBlendshapeBlinkDetector(opts);
    var last = { mode: null, value: 0 };
    return {
      update: function (result) {
        if (!result) return last;
        if (result.source === "face-api" && result.landmarks && result.landmarks.raw) {
          last = { mode: "ear", value: ear.update(result.landmarks.raw) };
        } else if (result.blink != null) {
          last = { mode: "blink", value: shape.update(result.blink) };
        }
        return last;
      },
      // 表示用のラベル（EARは開眼度、blinkは閉じ具合なので意味が逆）
      label: function () {
        if (last.mode === "ear") return "EAR " + last.value.toFixed(2);
        if (last.mode === "blink") return "まばたき " + last.value.toFixed(2);
        return "";
      },
      ok: function () { return ear.ok() || shape.ok(); },
      reset: function () { ear.reset(); shape.reset(); }
    };
  }

  global.FaceEngine = {
    createReader: createReader,
    createLiveness: createLiveness,
    createMediaPipe: createMediaPipe,
    describeAligned: describeAligned,
    alignRect: alignRect,
    createBlendshapeBlinkDetector: createBlendshapeBlinkDetector,
    MP_LEFT_EYE: MP_LEFT_EYE,
    MP_RIGHT_EYE: MP_RIGHT_EYE,
    MP_LIPS: MP_LIPS,
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
