import DefaultTheme from 'vitepress/theme'
import type { Theme } from 'vitepress'
import { h } from 'vue'
import VramEstimator from './components/VramEstimator.vue'
import ResearchHome from './components/ResearchHome.vue'
import './custom.css'
import './diagrams.css'

export default {
  extends: DefaultTheme,
  Layout() {
    return h(DefaultTheme.Layout, null, {
      'home-hero-before': () => h(ResearchHome)
    })
  },
  enhanceApp({ app }) {
    app.component('VramEstimator', VramEstimator)
  }
} satisfies Theme
