const {themes} = require('prism-react-renderer');

const repository = process.env.GITHUB_REPOSITORY || 'Muqi1029/ai-infra-bench';
const [organizationName, projectName] = repository.split('/');

/** @type {import('@docusaurus/types').Config} */
module.exports = {
  title: 'AI Infra Bench',
  tagline: 'Benchmark, evaluate, and size your inference service',
  url: process.env.DOCS_URL || `https://${organizationName.toLowerCase()}.github.io`,
  baseUrl: process.env.DOCS_BASE_URL || `/${projectName}/`,
  organizationName,
  projectName,
  trailingSlash: true,
  onBrokenLinks: 'throw',
  markdown: {hooks: {onBrokenMarkdownLinks: 'throw'}},
  i18n: {
    defaultLocale: 'zh-Hans',
    locales: ['zh-Hans', 'en'],
    localeConfigs: {
      'zh-Hans': {label: '简体中文', htmlLang: 'zh-CN'},
      en: {label: 'English', htmlLang: 'en'},
    },
  },
  presets: [['classic', {
    docs: {
      routeBasePath: '/',
      sidebarPath: require.resolve('./sidebars.js'),
      editUrl: ({locale, docPath}) => {
        const path = locale === 'en'
          ? `i18n/en/docusaurus-plugin-content-docs/current/${docPath}`
          : `docs/${docPath}`;
        return `https://github.com/${repository}/edit/main/website/${path}`;
      },
    },
    blog: false,
    theme: {customCss: require.resolve('./src/css/custom.css')},
  }]],
  themeConfig: {
    navbar: {
      title: 'AI Infra Bench',
      items: [
        {type: 'docSidebar', sidebarId: 'docs', label: '文档', position: 'left'},
        {type: 'localeDropdown', position: 'right'},
        {href: `https://github.com/${repository}`, label: 'GitHub', position: 'right'},
        {href: 'https://pypi.org/project/ai-infra-bench/', label: 'PyPI', position: 'right'},
      ],
    },
    footer: {
      style: 'dark',
      copyright: 'AI Infra Bench · Apache-2.0',
    },
    prism: {
      theme: themes.github,
      darkTheme: themes.dracula,
      additionalLanguages: ['bash', 'json', 'yaml', 'python'],
    },
  },
};
