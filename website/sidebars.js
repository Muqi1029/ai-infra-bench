module.exports = {
  docs: [
    'intro',
    'quickstart',
    {type: 'category', label: '使用指南', items: [
      'guides/requests',
      'guides/benchmark',
      'guides/slo',
      'guides/sessions',
      'guides/evaluation',
      'guides/monitoring',
    ]},
    {type: 'category', label: '参考', items: [
      'reference/metrics',
      'reference/cli',
      'troubleshooting',
    ]},
  ],
};
