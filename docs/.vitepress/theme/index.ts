import DefaultTheme from 'vitepress/theme'
import type { Theme } from 'vitepress'
import VramEstimator from './components/VramEstimator.vue'
import './custom.css'

export default {
  extends: DefaultTheme,
  enhanceApp({ app }) {
    app.component('VramEstimator', VramEstimator)
  }
} satisfies Theme
