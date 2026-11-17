# AIB Documentation (`website/`)

Built with [VitePress](https://vitepress.dev/). Simplified Chinese (`/`) is the default locale; English (`/en/`) is built into the same static bundle (`website/build`).

## Local Preview

Requires Node.js 20+ (CI uses Node.js 22):

```bash
cd website
npm ci
npm start        # http://localhost:5173/ai-infra-bench/
```

Build and preview the production output (`website/build`):

```bash
npm run build
npm run serve
```

## Layout

- Chinese pages: `docs/*.md`
- English pages: `docs/en/*.md`
- Site config & i18n navigation: `docs/.vitepress/config.mjs`
- Theme & custom Mintlify-style CSS: `docs/.vitepress/theme/`

Run `npm run build` before pushing. VitePress rejects dead internal links automatically.

## GitHub Pages

The workflow in [`.github/workflows/docs.yml`](../.github/workflows/docs.yml) runs `npm ci && npm run build` and deploys `website/build` on pushes to `main`:

- Chinese: `https://muqi1029.github.io/ai-infra-bench/`
- English: `https://muqi1029.github.io/ai-infra-bench/en/`

Override `GITHUB_REPOSITORY` or `DOCS_BASE_URL` (e.g. `/`) when hosting on a custom domain. Keep `package-lock.json` committed for reproducible CI builds.
