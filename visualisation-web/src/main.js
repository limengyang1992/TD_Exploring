import './assets/main.css'

import { createApp } from 'vue'
import VueLatex from 'vatex'
import { createPinia } from 'pinia'

import App from './App.vue'
import router from './router'

const app = createApp(App)

app.use(createPinia())
app.use(router)
app.use(VueLatex)

app.mount('#app')
