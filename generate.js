import 'dotenv/config';
import Replicate from 'replicate';
import { readFile, writeFile } from 'fs/promises';

const replicate = new Replicate();
const sleep = (ms) => new Promise(r => setTimeout(r, ms));

async function streamToFile(item, filename) {
  const chunks = [];
  for await (const chunk of item) chunks.push(chunk);
  await writeFile(filename, Buffer.concat(chunks));
}

async function runWithRetry(model, input, tries = 5) {
  for (let i = 0; i < tries; i++) {
    try {
      return await replicate.run(model, { input });
    } catch (e) {
      if (String(e).includes("429") && i < tries - 1) {
        console.log("Rate limited, waiting 5s...");
        await sleep(5000);
      } else throw e;
    }
  }
}

const imageBuffer = await readFile("room-real.jpg");
const imageDataUri = `data:image/jpeg;base64,${imageBuffer.toString("base64")}`;

console.log("Step 1/2: masking furniture...");
const maskOutput = await runWithRetry(
  "schananas/grounded_sam:ee871c19efb1941f55f66a3d7d960428c8a5afcb77449547fe8e5a3ab9ebc21c",
  { image: imageDataUri, mask_prompt: "furniture", negative_mask_prompt: "", adjustment_factor: 0 }
);
const maskItem = (Array.isArray(maskOutput) ? maskOutput : [maskOutput])[0];
await streamToFile(maskItem, "mask-bw.png");
const maskBuffer = await readFile("mask-bw.png");
const maskDataUri = `data:image/png;base64,${maskBuffer.toString("base64")}`;

await sleep(5000);

console.log("Step 2/2: generating furnished result (FLUX)...");
const genOutput = await runWithRetry(
  "zsxkib/flux-dev-inpainting:ca8350ff748d56b3ebbd5a12bd3436c2214262a4ff8619de9890ecc41751a008",
  {
    image: imageDataUri,
    mask: maskDataUri,
    prompt: "a beautifully staged modern living room, stylish grey sofa, wooden coffee table, soft cushions, warm natural lighting, photorealistic real estate listing photo",
    num_inference_steps: 28,
    strength: 0.75
  }
);
const genItem = (Array.isArray(genOutput) ? genOutput : [genOutput])[0];
await streamToFile(genItem, "result.png");

console.log("Done! Compare room-real.jpg (before) with result.png (after).");
