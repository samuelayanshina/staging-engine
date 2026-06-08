import 'dotenv/config';
import Replicate from 'replicate';
import { readFile, writeFile } from 'fs/promises';

const replicate = new Replicate();
const sleep = (ms) => new Promise(r => setTimeout(r, ms));

async function streamToBuffer(item) {
  const chunks = [];
  for await (const chunk of item) chunks.push(chunk);
  return Buffer.concat(chunks);
}
async function runWithRetry(model, input, tries = 6) {
  for (let i = 0; i < tries; i++) {
    try { return await replicate.run(model, { input }); }
    catch (e) {
      if (String(e).includes("429") && i < tries - 1) { console.log("  rate limited, waiting 8s..."); await sleep(8000); }
      else throw e;
    }
  }
}

const imgBuf = await readFile("room3-real.jpg");
const imgUri = `data:image/jpeg;base64,${imgBuf.toString("base64")}`;

console.log("masking...");
const maskOut = await runWithRetry(
  "schananas/grounded_sam:ee871c19efb1941f55f66a3d7d960428c8a5afcb77449547fe8e5a3ab9ebc21c",
  { image: imgUri, mask_prompt: "sofa, couch, coffee table, chair", negative_mask_prompt: "wall, window, lamp, picture, rug, floor", adjustment_factor: 5 }
);
const arr = Array.isArray(maskOut) ? maskOut : [maskOut];
// CORRECT mask = white furniture on black bg (index 2), NOT the green preview (index 0)
const maskBuf = await streamToBuffer(arr[2]);
await writeFile("used-mask.png", maskBuf);  // save so we can verify
const maskUri = `data:image/png;base64,${maskBuf.toString("base64")}`;

await sleep(8000);

console.log("staging with flux-fill-pro...");
const genOut = await runWithRetry(
  "black-forest-labs/flux-fill-pro",
  {
    image: imgUri,
    mask: maskUri,
    prompt: "a beautifully staged modern living room, stylish sofa, wooden coffee table, soft cushions, warm natural lighting, photorealistic real estate listing photo",
    steps: 50,
    guidance: 30
  }
);
const genBuf = await streamToBuffer((Array.isArray(genOut) ? genOut : [genOut])[0]);
await writeFile("pro-test2.png", genBuf);
console.log("Done! Check used-mask.png (should be white furniture on black), then compare room3-real.jpg with pro-test2.png");
