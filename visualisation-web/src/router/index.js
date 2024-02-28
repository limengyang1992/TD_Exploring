import { createRouter, createWebHashHistory } from 'vue-router'

const router = createRouter({
  history: createWebHashHistory(import.meta.env.BASE_URL),
  routes: [
    {
      path: '/',
      name: 'home',
      component: () => import('../views/home/homeView.vue')
    },
    {
      path: '/perception',
      name: 'perception',
      component: () => import('../views/perception/perceptionView.vue'),
    },
    {
      path: '/diagnosis',
      name: 'diagnosis',
      component: () => import('../views/diagnosis/diagnosisView.vue')
    },
    {
      path: '/optimize',
      name: 'optimize',
      component: () => import('../views/optimize/optimizeView.vue')
    },
    {
      path: '/contrast',
      name: 'contrast',
      component: () => import('../views/dnnContrast/dnnContrast.vue')
    },
    {
      path: '/extract',
      name: 'extract',
      component: () => import('../views/dnnExtract/dnnExtract.vue')
    },
    {
      path: '/detection',
      name: 'detection',
      component: () => import('../views/dnnDetection/dnnDetection.vue')
    },
    {
      path: '/learning',
      name: 'learning',
      component: () => import('../views/dnnLearning/dnnLearning.vue')
    }
  ]
})

export default router
