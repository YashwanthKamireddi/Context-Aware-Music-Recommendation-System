import { cp, mkdir, rm, stat } from "fs/promises";
import path from "path";
import { fileURLToPath } from "url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const projectRoot = path.resolve(__dirname, "..");
const sourceDir = path.join(projectRoot, "frontend");
const distDir = path.join(projectRoot, "dist");

async function ensureExists(dir) {
  try {
    await stat(dir);
    return true;
  } catch (error) {
    return false;
  }
}

async function main() {
  const hasFrontend = await ensureExists(sourceDir);
  if (!hasFrontend) {
    throw new Error(`Missing frontend directory at ${sourceDir}`);
  }

  await rm(distDir, { recursive: true, force: true });
  await mkdir(distDir, { recursive: true });
  await cp(sourceDir, distDir, { recursive: true });

  console.log(`Copied frontend assets to ${distDir}`);
}

main().catch((error) => {
  console.error("Failed to prepare frontend bundle:", error);
  process.exit(1);
});
