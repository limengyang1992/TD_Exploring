<script setup>
import { RouterView } from 'vue-router'
import navHeader from './components/navHeader.vue'
// import rootFooter from './components/footer.vue';
</script>

<template>
    
<navHeader />
      
  <RouterView />
  <!-- <div class="fotter"></div> -->
  <!-- <rootFooter /> -->

</template>

<style scoped>
/* .fotter{
  width: 100%;
  height: 190px;
} */
</style>
