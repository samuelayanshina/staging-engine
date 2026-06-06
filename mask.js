import 'dotenv/config';
import Replicate from 'replicate';
import { readFile, writeFile } from 'fs/promises';

const replicate = new Replicate();

console.log("Masking the room...");

const imageBuffer = await readFile("room-real.jpg");
const imageDataUri = `data:image/jpeg;base64,${imageBuffer.toString("base64")}`;

const output = await replicate.run(
  "schananas/grounded_sam:ee871c19efb1941f55f66a3d7d960428c8a5afcb77449547fe8e5a3ab9ebc21c",
  {
    input: {
      image: imageDataUri,
      mask_prompt: "furniture",
      negative_mask_prompt: "",
      adjustment_factor: 0
    }
  }
);

console.log("Mask output:", output);

const masks = Array.isArray(output) ? output : [output];
const res = await fetch(masks[0]);
const buf = Buffer.from(await res.arrayBuffer());
await writeFile("mask.png", buf);
console.log("Saved mask.png — open it to check.");
