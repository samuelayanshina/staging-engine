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
      if (String(e).includes("429") && i < tries - 1) { console.log("  rate limited, waiting 10s..."); await sleep(10000); }
      else throw e;
    }
  }
}

const file = "empty1-real.jpg";
const imgBuf = await readFile(file);
const imgUri = `data:image/jpeg;base64,${imgBuf.toString("base64")}`;

console.log("masking floor + lower wall area...");
const maskOut = await runWithRetry(
  "schananas/grounded_sam:ee871c19efb1941f55f66a3d7d960428c8a5afcb77449547fe8e5a3ab9ebc21c",
  { image: imgUri, mask_prompt: "floor, lower wall, room interior space", negative_mask_prompt: "ceiling, window", adjustment_factor: 15 }
);
const arr = Array.isArray(maskOut) ? maskOut : [maskOut];
const maskBuf = await streamToBuffer(arr[2]);
await writeFile("stage-mask3.png", maskBuf);
const maskUri = `data:image/png;base64,${maskBuf.toString("base64")}`;

await sleep(10000);

console.log("staging a FULL furniture set...");
const genOut = await runWithRetry(
  "black-forest-labs/flux-fill-pro",
  {
    image: imgUri,
    mask: maskUri,
    prompt: "a fully furnished modern living room, large grey sofa against the wall, two armchairs, wooden coffee table, large area rug, side tables with lamps, wall art, plants, fully styled and professionally staged, photorealistic real estate listing photo, bright natural lighting, furniture filling the room",
    steps: 50,
    guidance: 30
  }
);
const genBuf = await streamToBuffer((Array.isArray(genOut) ? genOut : [genOut])[0]);
await writeFile("staged-test3.png", genBuf);
console.log("Done! Check stage-mask3.png (white should now cover floor AND up the walls), then compare empty1-real.jpg with staged-test3.png");
