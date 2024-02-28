<template>
    <div class="footer">
        <div class="footer_box">
            <span>Copyright@ All Rights Reserved 津ICP备2024011494号</span>
        <div>
            <template v-for="item in cardList" :key="item">
            <a :href="item.url" target="_blank" rel="noopener noreferrer">
                <img :src="item.img" alt="">
            </a>
        </template>
        </div>
        </div>
    </div>
</template>
<script setup>
import { ref } from 'vue'
const cardList = ref([
  {
    img: new URL('../assets/tuandui.png', import.meta.url).href,
    url: 'http://tj.teacher.360eol.com/teacherBasic/preview?teacherId=2933'
  },
  {
    img: new URL('../assets/github.png', import.meta.url).href,
    url: 'https://github.com/limengyang1992/TD_Exploring'
  },
  {
    img: new URL('../assets/baidu.png', import.meta.url).href,
    url: 'https://pan.baidu.com/s/1eRbFdVt9DXlV-N-ukttmLg?pwd=1sh9'
  }
])
</script>
<style scoped>
.footer {
    min-height: 190px;
    background-color: #202D42;
    color: #808795;
    font-size: 14px;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: flex-start;
}
.footer_box {
    width: 1400px;
    padding: 0 200px;
}
a {
    display: inline-block;
    width: 40px;
    height: 40px;
    margin-top: 20px;
    padding-right: 20px;
    pointer-events:none;
}
a img {
    width: 100%;
}
</style>