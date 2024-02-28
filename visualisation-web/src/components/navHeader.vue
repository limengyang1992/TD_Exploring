

<template>
  <div class="header">
    <div class="logo"><img src="../assets/logo.png" alt=""></div>
    <el-menu class="el-menu-demo" mode="horizontal" @select="handleSelect">
      <el-menu-item index="/home">
        <RouterLink to="/">Home</RouterLink>
      </el-menu-item>
      <el-sub-menu index="/perceptions">
        <template #title>Perception</template>
        <el-menu-item index="/extract">
          <RouterLink to="">TD Quantities</RouterLink>
        </el-menu-item>
        <el-menu-item index="/perception">
          <RouterLink to="/diagnosis">TD Visualization</RouterLink>
        </el-menu-item>
        <!-- <el-menu-item index="/contrast">
          <RouterLink to="contrast">TD Compare</RouterLink>
        </el-menu-item> -->
        <el-menu-item index="/detection">
          <RouterLink to="detection">Noise detection</RouterLink>
        </el-menu-item>
        <el-menu-item index="/learning">
          <RouterLink to="learning">Imbalance learning</RouterLink>
        </el-menu-item>
      </el-sub-menu>
      <el-menu-item index="/diagnosis">
        <RouterLink to="/diagnosis">Diagnosis</RouterLink>
      </el-menu-item>
      <el-menu-item index="/optimize">
        <RouterLink to="/optimize">Optimization</RouterLink>
      </el-menu-item>
    </el-menu>
  </div>
</template>
<script setup>
import { useRouter } from 'vue-router'
const { push } = useRouter()
const handleSelect = (key, keyPath) => {
  console.log(key, keyPath)
  if (keyPath.length == 2) {
    push(keyPath[1])
  }
}
</script>
<style scoped>
a {
  text-decoration: none;
  color: #081642;
  transition: 0.4s;
  padding: 0 10px;
  border-radius: 6px;
}

.header {
  border: none;
  background-color: #fff;
  z-index: 10000;
  -webkit-transition: background-color 0.5s linear;
  transition: background-color 0.5s linear;
  -webkit-box-shadow: 0 0 20px rgba(0, 0, 0, 0.2);
  box-shadow: 0 0 20px rgba(0, 0, 0, 0.2);
  padding: 0 60px;
  display: flex;
}
.logo {
  display: flex;
  align-items: center;
  padding-right: 10px;
}
.logo > img {
  width: 40px;
  height: 40px;
  border-radius: 50%;
  /* background-color: #ccc; */
  margin-right: 20px;
}
:deep(.el-menu--horizontal.el-menu) {
  border: none;
  min-width: 1000px;
  display: flex;
  /* justify-content: center; */
  align-items: center;
  border-bottom: none;
}

:deep(.el-menu--horizontal>.el-menu-item),
:deep(.el-menu--horizontal>.el-sub-menu) {
  position: relative;
  display: inline-block;
  color: #333;
}

:deep(.el-menu--horizontal>.el-menu-item::before),
:deep(.el-menu--horizontal>.el-sub-menu::before) {
  content: "";
  position: absolute;
  width: 100%;
  height: 4px;
  background-color: #333;
  top: -2px;
  left: 0;
  transform: scaleX(0);
  transform-origin: center;
  transition: transform 0.3s ease;
  box-sizing: border-box;
}

:deep(.el-menu--horizontal>.el-menu-item:hover::before),
:deep(.el-menu--horizontal>.el-sub-menu:hover::before) {
  transform: scaleX(1);
}

:deep(.el-menu--horizontal>.el-menu-item::after),
:deep(.el-menu--horizontal>.el-sub-menu::after) {
  content: "";
  position: absolute;
  width: 100%;
  height: 4px;
  background-color: #333;
  top: -2px;
  right: 0;
  transform: scaleX(0);
  transform-origin: center;
  transition: transform 0.3s ease;
}

:deep(.el-menu--horizontal>.el-menu-item:hover::after),
:deep(.el-menu--horizontal>.el-sub-menu:hover::after) {
  transform: scaleX(1);
}
:deep(.el-menu--horizontal>.el-menu-item:hover),
:deep(.el-menu--horizontal>.el-sub-menu:hover) {
  background-color: rgba(249,249,249);
  color: #333;
}
:deep(.el-menu--horizontal>.el-menu-item.is-active),
:deep(.el-menu--horizontal>.el-sub-menu.is-active .el-sub-menu__title) {
  border-bottom: none;
  border-top: 3px solid #F8466F;
  color: #F8466F !important;
}
:deep(.el-menu--horizontal>.el-menu-item.is-active::after),
:deep(.el-menu--horizontal>.el-sub-menu.is-active::after) {
  content: "";
  position: absolute;
  width: 100%;
  height: 2px;
  background-color: #F8466F;
  top: -2px;
  right: 0;
}
:deep(.el-menu--horizontal>.el-menu-item.is-active::before),
:deep(.el-menu--horizontal>.el-sub-menu.is-active::before) {
  content: "";
  position: absolute;
  width: 100%;
  height: 2px;
  background-color: #F8466F;
  top: -2px;
  right: 0;
}
</style>
