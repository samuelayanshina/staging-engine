import 'dotenv/config';
import Replicate from 'replicate';

const replicate = new Replicate();

console.log("Calling Replicate...");

const output = await replicate.run(
  "black-forest-labs/flux-schnell",
  { input: { prompt: "a simple red apple on a white table" } }
);

console.log("It works. Output:", output);
