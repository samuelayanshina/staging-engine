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
    try {
      return await replicate.run(model, { input });
    } catch (e) {
      if (String(e).includes("429") && i < tries - 1) {
        console.log("  rate limited, waiting 8s...");
        await sleep(8000);
      } else throw e;
    }
  }
}

const rooms = ["room1-real.jpg", "room2-real.jpg", "room3-real.jpg", "room4-real.jpg"];

for (let n = 0; n < rooms.length; n++) {
  const file = rooms[n];
  console.log(`\nRoom ${n + 1}/4: ${file}`);

  const imgBuf = await readFile(file);
  const imgUri = `data:image/jpeg;base64,${imgBuf.toString("base64")}`;

  console.log("  masking...");
  const maskOut = await runWithRetry(
    "schananas/grounded_sam:ee871c19efb1941f55f66a3d7d960428c8a5afcb77449547fe8e5a3ab9ebc21c",
    { image: imgUri, mask_prompt: "furniture", negative_mask_prompt: "", adjustment_factor: 0 }
  );
  const maskBuf = await streamToBuffer((Array.isArray(maskOut) ? maskOut : [maskOut])[0]);
  const maskUri = `data:image/png;base64,${maskBuf.toString("base64")}`;

  await sleep(8000);

  console.log("  staging...");
  const genOut = await runWithRetry(
    "zsxkib/flux-dev-inpainting:ca8350ff748d56b3ebbd5a12bd3436c2214262a4ff8619de9890ecc41751a008",
    {
      image: imgUri,
      mask: maskUri,
      prompt: "a beautifully staged modern living room, stylish sofa, wooden coffee table, soft cushions, warm natural lighting, photorealistic real estate listing photo",
      num_inference_steps: 28,
      strength: 0.75
    }
  );
  const genBuf = await streamToBuffer((Array.isArray(genOut) ? genOut : [genOut])[0]);
  await writeFile(`after-${n + 1}.png`, genBuf);
  console.log(`  saved after-${n + 1}.png`);

  await sleep(8000);
}

console.log("\nDone! Compare each roomN-real.jpg (before) with after-N.png (after).");
