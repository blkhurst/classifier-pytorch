import { fileURLToPath, URL } from "node:url";
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

// The Browser Uses One Origin; Vite Forwards API Requests To Python.
const proxy = { "/api": "http://127.0.0.1:8000" };

export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: { alias: { "@": fileURLToPath(new URL("./src", import.meta.url)) } },
  server: { proxy, port: 5173, strictPort: true },
  preview: { proxy },
});
