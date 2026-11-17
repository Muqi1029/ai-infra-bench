import { defineConfig } from 'vitepress';

const repository = process.env.GITHUB_REPOSITORY || 'Muqi1029/ai-infra-bench';
const [, projectName = 'ai-infra-bench'] = repository.split('/');
const base = process.env.DOCS_BASE_URL || `/${projectName}/`;

export default defineConfig({
  title: 'AI Infra Bench',
  base,
  outDir: '../build',
  cacheDir: '../.vitepress/cache',
  cleanUrls: true,
  lastUpdated: false,

  head: [
    ['link', { rel: 'icon', type: 'image/svg+xml', href: `${base}logo.svg` }],
  ],

  markdown: {
    theme: {
      light: 'github-light',
      dark: 'github-dark',
    },
  },

  themeConfig: {
    logo: '/logo.svg',
    siteTitle: 'AI Infra Bench',

    search: {
      provider: 'local',
      options: {
        locales: {
          root: {
            translations: {
              button: {
                buttonText: '搜索文档',
                buttonAriaLabel: '搜索文档',
              },
              modal: {
                noResultsText: '未找到相关内容',
                resetButtonTitle: '清除查询条件',
                footer: {
                  selectText: '选择',
                  navigateText: '切换',
                  closeText: '关闭',
                },
              },
            },
          },
        },
      },
    },

    socialLinks: [
      { icon: 'github', link: `https://github.com/${repository}` },
    ],

    footer: {
      message: 'Released under the Apache-2.0 License.',
      copyright: 'AI Infra Bench (aib)',
    },
  },

  locales: {
    root: {
      label: '简体中文',
      lang: 'zh-CN',
      description: '轻量级推理压测、正确性评测与 SLO 容量探测工具',
      themeConfig: {
        nav: [
          { text: '快速开始', link: '/quickstart' },
          { text: '压测与 SLO', link: '/benchmark' },
          { text: '正确性评测', link: '/evaluation' },
          { text: '工具集', link: '/tools' },
          { text: '参考', link: '/reference' },
          { text: 'PyPI', link: 'https://pypi.org/project/ai-infra-bench/' },
        ],
        sidebar: [
          {
            text: '使用指南',
            items: [
              { text: '快速开始', link: '/quickstart' },
              { text: '压测与 SLO', link: '/benchmark' },
              { text: '正确性评测', link: '/evaluation' },
              { text: '请求与诊断工具', link: '/tools' },
              { text: '命令与指标参考', link: '/reference' },
            ],
          },
        ],
        outline: {
          label: '本页目录',
          level: [2, 3],
        },
        docFooter: {
          prev: '上一页',
          next: '下一页',
        },
        returnToTopLabel: '回到顶部',
        sidebarMenuLabel: '菜单',
        darkModeSwitchLabel: '外观',
      },
    },

    en: {
      label: 'English',
      lang: 'en',
      link: '/en/',
      description: 'Lightweight CLI for LLM serving benchmarks, correctness evaluation, and SLO sizing',
      themeConfig: {
        nav: [
          { text: 'Quickstart', link: '/en/quickstart' },
          { text: 'Benchmark', link: '/en/benchmark' },
          { text: 'Evaluation', link: '/en/evaluation' },
          { text: 'Tools', link: '/en/tools' },
          { text: 'Reference', link: '/en/reference' },
          { text: 'PyPI', link: 'https://pypi.org/project/ai-infra-bench/' },
        ],
        sidebar: [
          {
            text: 'Guide',
            items: [
              { text: 'Quickstart', link: '/en/quickstart' },
              { text: 'Benchmark & SLO', link: '/en/benchmark' },
              { text: 'Evaluation', link: '/en/evaluation' },
              { text: 'Requests & Tools', link: '/en/tools' },
              { text: 'CLI & Metrics', link: '/en/reference' },
            ],
          },
        ],
        outline: {
          label: 'On this page',
          level: [2, 3],
        },
      },
    },
  },
});
