<template>
    <div class="home">
      <div class="content">
        <div class="title">
          <p>In Planning: Optimization</p>
          <div class="des">
            <div>The third phase of our research plan focuses on optimizing the training process. The initial strategy involves creating an AI Agent to facilitate this optimization, encompassing tasks such as selecting optimal hyperparameters (e.g., learning rate and epoch) and choosing effective learning strategies, including sample weighting and perturbation strategies.</div>
  
          </div>
        </div>
        <div class="homeEcharts">
          <homeEcharts></homeEcharts>
        </div>
      </div>
    </div>
  </template>
  <script setup>
  import homeEcharts from '../../components/homeEcharts.vue'
  </script>
  <style scoped>
  .home {
    max-width: 1400px !important;
    min-height: calc(100vh - 100px);
    margin: 0 auto;
    padding-top: 100px;
  }
  
  /* .content {
    display: flex;
    justify-content: center;
    align-items: center;
  } */
  
  .content>div {
    flex: 1;
  }
  .homeEcharts {
    display: flex;
    justify-content: center;
  }
  .content>div.title {
    padding: 0 100px;
  }
  
  .title>p {
    font-size: 50px;
    font-weight: 800;
    color: #081642;
    line-height: 1.2em;
    padding-bottom: 60px;
  }
  
  .des {
    display: flex;
  }
  
  .des div {
    font-size: 20px;
    text-align: justify;
    flex: 1;
    display: flex;
    justify-content: center;
    align-items: center;
  }
  
  .aboutUs {
    padding-bottom: 165px;
  }
  
  .aboutUs_title {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding-top: 50px;
  }
  
  .aboutUs_title h3 {
    font-weight: 600;
    font-size: 35px;
    color: #06133B;
  }
  
  .aboutUs_title hr {
    display: inline-block;
    text-align: center;
    width: 36px;
    height: 2px;
    background-color: #8EE3A9;
    color: #8EE3A9;
    margin: 24px 0;
    border: 0px;
  }
  
  .aboutUs_title span {
    font-size: 12px;
    color: #949CB1;
    letter-spacing: 0;
  }
  
  .aboutUs_list {
    padding: 0 100px;
    margin-top: 30px;
  }
  
  a {
    text-decoration: none;
    font-size: 20px;
    color: #06133B;
  }
  
  .card_item {
    display: flex;
    align-items: center;
  }
  
  .card_item img {
    width: 40px;
    height: 40px;
    margin-right: 14px;
  }
  
  .card_item span:nth-child(2) {
    flex: 1;
  }
  
  .card_item .icon-v-right {
    display: inline-block;
    width: 12px;
    height: 12px;
    border: 2px solid rgba(148, 156, 177, 0.4);
    border-width: 2px 2px 0 0;
    -webkit-transform: rotate(45deg);
    -ms-transform: rotate(45deg);
    transform: rotate(45deg);
    margin: 0;
    line-height: 60px;
  }
  :deep(.el-row) {
    justify-content: center;
  }
  </style>
  