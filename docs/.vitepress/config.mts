import { defineConfig } from 'vitepress'

export default defineConfig({
  title: 'AI Toolkit',
  description: 'Training configuration and workflow guide for AI Toolkit',
  cleanUrls: true,
  themeConfig: {
    nav: [
      { text: 'Guide', link: '/guide/getting-started' },
      { text: 'Configuration', link: '/guide/configuration' },
      { text: 'Optimizers', link: '/guide/optimizers' },
      { text: 'Models', link: '/guide/models' },
    ],
    sidebar: [
      {
        text: 'Training Guide',
        items: [
          { text: 'Getting Started', link: '/guide/getting-started' },
          { text: 'Configuration', link: '/guide/configuration' },
          { text: 'Training Process', link: '/guide/training-process' },
          { text: 'Optimizers', link: '/guide/optimizers' },
          { text: 'Model Notes', link: '/guide/models' },
          { text: 'Troubleshooting', link: '/guide/troubleshooting' },
        ],
      },
    ],
    socialLinks: [
      { icon: 'github', link: 'https://github.com/ostris/ai-toolkit' },
    ],
    search: { provider: 'local' },
    editLink: {
      pattern: 'https://github.com/ostris/ai-toolkit/edit/main/docs/:path',
    },
  },
})
