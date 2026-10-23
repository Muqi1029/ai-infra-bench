# AIB documentation / 用户文档站

Built with Docusaurus 3 and MDX. Simplified Chinese is the default locale; English is available through the language menu. Both locales are built into one static site.

## Local development / 本地预览

Requires Node.js 20+ and npm (CI uses Node.js 22).

```bash
cd website
npm ci
npm start                 # Chinese / 中文
npm start -- --locale en  # English / 英文
```

For both locales, build and serve the static site:

```bash
npm run build
npm run serve
```

Default local URL: `http://localhost:3000/ai-infra-bench/`. English: `http://localhost:3000/ai-infra-bench/en/`.

## Editing / 内容维护

- Chinese source: `docs/**/*.mdx`.
- English source: `i18n/en/docusaurus-plugin-content-docs/current/**/*.mdx`; keep document IDs and relative paths identical to Chinese.
- Navigation: `sidebars.js`; English category labels: `i18n/en/docusaurus-plugin-content-docs/current.json`.
- English navigation labels: `i18n/en/docusaurus-theme-classic/navbar.json`.
- Site settings: `docusaurus.config.js`; styles: `src/css/custom.css`.

Use front matter, fenced code blocks, relative `.mdx` links, and Docusaurus admonitions (`:::tip`, `:::info`). Standard Markdown works inside MDX. Do not add unescaped JSX-like text or braces outside code blocks. Update both translations when changing a command or example.

Run `npm run build` before submitting. Broken links fail the build. No Python service or model is required to build the website.

## GitHub Pages

In repository **Settings → Pages → Build and deployment**, set **Source** to **GitHub Actions**. After merging to `main`, the Documentation workflow builds both locales and deploys the site. Pull requests only build for validation.

在仓库 **Settings → Pages** 将发布来源设为 **GitHub Actions**。合并至 `main` 后自动构建并部署；PR 只验证构建。该设置需要仓库管理员操作，本地构建不代表已经上线。

Default published URLs for this repository:

- Chinese: `https://muqi1029.github.io/ai-infra-bench/`
- English: `https://muqi1029.github.io/ai-infra-bench/en/`

The workflow derives the owner/repository from `GITHUB_REPOSITORY`, including on forks. For a custom domain or a different hosting path, set `DOCS_URL` (origin only) and `DOCS_BASE_URL` (leading and trailing `/`) in the build environment and configure your hosting domain. User/organization Pages repositories (`owner.github.io`) should set `DOCS_BASE_URL=/`.

The package override pins webpack 5.101.3 for compatibility with Docusaurus 3.9.2's progress plugin. Review that override when upgrading Docusaurus; keep `package-lock.json` committed for reproducible CI builds.
