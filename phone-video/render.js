// phone.html をフレームごとに描き出し、ffmpeg で mp4 にする
// 使い方: node render.js [fps]
const { chromium } = require('/opt/node22/lib/node_modules/playwright');
const { spawn } = require('child_process');
const path = require('path');

(async () => {
  const fps = Number(process.argv[2] || 30);
  const browser = await chromium.launch();
  const page = await browser.newPage({ viewport: { width: 1080, height: 1920 } });
  await page.goto('file://' + path.join(__dirname, 'phone.html') + '?render');
  await page.evaluate(() => document.fonts.load('700 64px IPAGothic'));
  const duration = await page.evaluate(() => window.DURATION);
  const out = path.join(__dirname, 'phone.mp4');
  const ff = spawn('ffmpeg', ['-y', '-f', 'image2pipe', '-framerate', String(fps), '-i', '-',
    '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '18', '-preset', 'slow', '-movflags', '+faststart', out],
    { stdio: ['pipe', 'ignore', 'inherit'] });
  const total = Math.round(duration * fps);
  for (let i = 0; i < total; i++) {
    const b64 = await page.evaluate(t => { renderFrame(t); return document.getElementById('c').toDataURL('image/png').slice(22); }, i / fps);
    if (!ff.stdin.write(Buffer.from(b64, 'base64'))) await new Promise(r => ff.stdin.once('drain', r));
    if (i % 60 === 0) console.log(`frame ${i}/${total}`);
  }
  ff.stdin.end();
  await new Promise(r => ff.on('close', r));
  await browser.close();
  console.log('done:', out);
})();
