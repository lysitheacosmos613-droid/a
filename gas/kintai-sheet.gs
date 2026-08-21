/**
 * 勤怠システム（kintai.html）→ Googleスプレッドシート 連携用スクリプト
 *
 * 使い方は kintai.md の「Googleスプレッドシート連携」を参照。
 * ざっくり言うと：スプレッドシートを作る → 拡張機能 → Apps Script → このコードを貼る
 * → 下の TOKEN を自分で決めた合言葉に書き換える → デプロイ（ウェブアプリ）→ URLをiPadに登録。
 *
 * 打刻ごとに「打刻」シートへ1行追加し、そのたびに「日別」シートを作り直します。
 */

// ★ここを自分で決めた合言葉に変えてください（iPad側にも同じものを入力します）
var TOKEN = "kintai-himitsu-2026";

var SHEET_PUNCH = "打刻";
var SHEET_DAILY = "日別";
var HEADER_PUNCH = ["打刻ID", "日時", "氏名", "種別", "方法", "端末", "受信時刻"];
var HEADER_DAILY = ["日付", "氏名", "出勤", "退勤", "実働(時:分)", "実働(分)", "備考"];

function doPost(e) {
  try {
    var body = JSON.parse(e.postData.contents);
    if (body.token !== TOKEN) return json({ ok: false, error: "合言葉が違います" });

    if (body.ping) {
      return json({ ok: true, pong: true, total: countPunches(), sheet: SpreadsheetApp.getActiveSpreadsheet().getName() });
    }

    var result = appendPunches(body.punches || [], body.device || "");
    rebuildDaily();
    return json({ ok: true, added: result.added, updated: result.updated, skipped: result.skipped, total: countPunches() });
  } catch (err) {
    return json({ ok: false, error: String(err) });
  }
}

// ブラウザで直接開いたときの動作確認用
function doGet(e) {
  var token = e && e.parameter ? e.parameter.token : "";
  if (token !== TOKEN) return json({ ok: false, error: "合言葉が違います" });
  return json({ ok: true, total: countPunches(), sheet: SpreadsheetApp.getActiveSpreadsheet().getName() });
}

function json(obj) {
  return ContentService.createTextOutput(JSON.stringify(obj)).setMimeType(ContentService.MimeType.JSON);
}

function sheetOf(name, header) {
  var ss = SpreadsheetApp.getActiveSpreadsheet();
  var sh = ss.getSheetByName(name);
  if (!sh) {
    sh = ss.insertSheet(name);
    sh.appendRow(header);
    sh.getRange(1, 1, 1, header.length).setFontWeight("bold");
    sh.setFrozenRows(1);
  }
  return sh;
}

function countPunches() {
  var sh = sheetOf(SHEET_PUNCH, HEADER_PUNCH);
  return Math.max(0, sh.getLastRow() - 1);
}

/**
 * 打刻IDをキーに追記する。同じIDが既にあれば、その行を上書きする
 * （iPad側で時刻を修正した打刻を送り直したときに反映されるように）。
 * 内容が変わっていなければ何もしない。
 */
function appendPunches(punches, device) {
  var sh = sheetOf(SHEET_PUNCH, HEADER_PUNCH);
  var rowOf = {};
  if (sh.getLastRow() > 1) {
    sh.getRange(2, 1, sh.getLastRow() - 1, 1).getValues().forEach(function (r, i) {
      if (r[0]) rowOf[String(r[0])] = i + 2;
    });
  }
  var newRows = [], updated = 0, skipped = 0;
  var now = new Date();
  punches.forEach(function (p) {
    if (!p || !p.id) { skipped++; return; }
    var row = [
      p.id,
      p.ts ? new Date(p.ts) : "",
      p.name || "",
      p.type === "in" ? "出勤" : "退勤",
      p.source === "manual" ? "手修正" : "顔認証",
      p.device || device || "",
      now
    ];
    var at = rowOf[p.id];
    if (at) {
      var cur = sh.getRange(at, 1, 1, HEADER_PUNCH.length).getValues()[0];
      var same = String(cur[1] && cur[1].getTime ? cur[1].getTime() : cur[1]) ===
                 String(row[1] && row[1].getTime ? row[1].getTime() : row[1]) &&
                 String(cur[2]) === String(row[2]) && String(cur[3]) === String(row[3]);
      if (same) { skipped++; return; }
      sh.getRange(at, 1, 1, HEADER_PUNCH.length).setValues([row]);
      updated++;
    } else {
      rowOf[p.id] = -1; // 同じ送信内に重複IDがあっても2行にしない
      newRows.push(row);
    }
  });
  if (newRows.length) {
    sh.getRange(sh.getLastRow() + 1, 1, newRows.length, HEADER_PUNCH.length).setValues(newRows);
  }
  if (sh.getLastRow() > 1) {
    sh.getRange(2, 2, sh.getLastRow() - 1, 1).setNumberFormat("yyyy/mm/dd hh:mm:ss");
    sh.getRange(2, 7, sh.getLastRow() - 1, 1).setNumberFormat("yyyy/mm/dd hh:mm:ss");
  }
  return { added: newRows.length, updated: updated, skipped: skipped };
}

/** 打刻シートから日別の勤務時間を組み立て直す */
function rebuildDaily() {
  var src = sheetOf(SHEET_PUNCH, HEADER_PUNCH);
  var out = sheetOf(SHEET_DAILY, HEADER_DAILY);
  var tz = Session.getScriptTimeZone();

  var values = src.getLastRow() > 1 ? src.getRange(2, 1, src.getLastRow() - 1, HEADER_PUNCH.length).getValues() : [];
  var byName = {};
  values.forEach(function (r) {
    var name = r[2], ts = r[1], type = r[3];
    if (!name || !(ts instanceof Date)) return;
    if (!byName[name]) byName[name] = [];
    byName[name].push({ ts: ts, type: type });
  });

  var rows = [];
  Object.keys(byName).forEach(function (name) {
    var list = byName[name].sort(function (a, b) { return a.ts - b.ts; });
    var open = null;
    list.forEach(function (p) {
      if (p.type === "出勤") {
        if (open) rows.push(row(name, open.ts, null, "退勤打刻なし"));
        open = p;
      } else {
        if (open) { rows.push(row(name, open.ts, p.ts, "")); open = null; }
        else rows.push(row(name, null, p.ts, "出勤打刻なし"));
      }
    });
    if (open) rows.push(row(name, open.ts, null, "勤務中"));
  });

  function row(name, inAt, outAt, warn) {
    var base = inAt || outAt;
    var minutes = (inAt && outAt) ? Math.round((outAt - inAt) / 60000) : "";
    return [
      Utilities.formatDate(base, tz, "yyyy/MM/dd"),
      name,
      inAt ? Utilities.formatDate(inAt, tz, "HH:mm") : "",
      outAt ? Utilities.formatDate(outAt, tz, "HH:mm") : "",
      minutes === "" ? "" : Math.floor(minutes / 60) + ":" + ("0" + (minutes % 60)).slice(-2),
      minutes,
      warn
    ];
  }

  rows.sort(function (a, b) { return a[0] < b[0] ? -1 : a[0] > b[0] ? 1 : (a[1] < b[1] ? -1 : 1); });

  if (out.getLastRow() > 1) out.getRange(2, 1, out.getLastRow() - 1, HEADER_DAILY.length).clearContent();
  if (rows.length) out.getRange(2, 1, rows.length, HEADER_DAILY.length).setValues(rows);
}
