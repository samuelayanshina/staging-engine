import 'dotenv/config';
import Replicate from 'replicate';
import { readFile, writeFile } from 'fs/promises';

const replicate = new Replicate();
async function streamToBuffer(item) {
  const chunks = [];
  for await (const chunk of item) chunks.push(chunk);
  return Buffer.concat(chunks);
}

const imgBuf = await readFile("room3-real.jpg");
const imgUri = `data:image/jpeg;base64,${imgBuf.toString("base64")}`;

console.log("getting mask...");
const out = await replicate.run(
  "schananas/grounded_sam:ee871c19efb1941f55f66a3d7d960428c8a5afcb77449547fe8e5a3ab9ebc21c",
  { input: { image: imgUri, mask_prompt: "sofa, couch, coffee table, chair", negative_mask_prompt: "wall, window, lamp, picture, rug, floor", adjustment_factor: 0 } }
);

const arr = Array.isArray(out) ? out : [out];
for (let i = 0; i < arr.length; i++) {
  const buf = await streamToBuffer(arr[i]);
  await writeFile(`rawmask-${i}.png`, buf);
  console.log(`saved rawmask-${i}.png`);
}
console.log("Open the rawmask files — we want crisp white furniture on black.");
