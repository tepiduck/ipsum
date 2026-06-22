import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Frontend reads static run artifacts from /runs/* (copied/symlinked into public/runs).
export default defineConfig({
  plugins: [react()],
  server: { port: 5173 },
});
