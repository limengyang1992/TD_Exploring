<template>
    <div class="home">
      <div class="content">
        <div class="title">
          <p>Perception-Diagnosis-Optimization</p>
          <div class="des">
            <div>Deep Neural Networks (DNNs) are rapidly outpacing alternative data processing techniques across various fields and applications. Despite their notable success in practical applications, DNNs still grapple with inadequate interpretability and explainability. Our project delves deeply into the training dynamics of samples, encompassing three key stages: training dynamics-based perception, training process diagnosis, and training process optimization. Building on the promising outcomes of our ongoing endeavors, we have secured funding for a new study, specifically targeting Stages 1 and 2, with a focus on large-scale deep-learning tasks. Our study is set to cover a substantial range, involving approximately 50,000 to 200,000 deep-learning tasks, contingent upon the availability of high-performance GPUs. Our analysis will be characterized by a meticulous consideration of a diverse array of factors. Our ultimate goal is to diagnose the unreasonable factors in the training process, paving the way for the development of an AI agent to enhance the optimization of the training procedure.</div>
  
          </div>
        </div>
        <div class="lineEchart">
          <template v-for="item in 6" :key="item">
            <div class="main" style="width: 400px;height: 300px;"></div>
          </template>
        </div>
        <div class="homeEcharts">
          <!-- <homeEcharts></homeEcharts> -->
        </div>
      </div>
      <div class="aboutUs">
        <div class="aboutUs_title">
          <h3>Open Source</h3>
          <hr>
          <span>If you have any questions, please contact me at: limengyang@tju.edu.cn.</span>
        </div>
        <div class="aboutUs_list">
          <el-row :gutter="80">
            <el-col v-for="(o) in cardList" :key="o" :span="8">
              <el-card :body-style="{ padding: '0px' }">
                <a :href="o.url" target="_blank" rel="noopener noreferrer">
                  <div style="padding: 14px" class="card_item">
                    <img :src="o.img" alt="">
                    <span class="spanLable">{{ o.label }}</span>
                    <span class="icon-v-right"></span>
                  </div>
                </a>
              </el-card>
            </el-col>
          </el-row>
        </div>
       
      </div>
     
    </div>
    <rootFooter></rootFooter>
  </template>
  <script setup>
  import { ref, onMounted, onBeforeUnmount } from 'vue'
  // import homeEcharts from '../../components/homeEcharts.vue'
  import * as echarts from 'echarts';
  import { home } from '../../services/data'
  import rootFooter from '../../components/footer.vue';
  const cardList = ref([
    {
      img: new URL('../../assets/github.png', import.meta.url).href,
      label: 'Github',
      url: 'https://github.com/limengyang1992/TD_Exploring'
    },
    {
      img: new URL('../../assets/baidu.png', import.meta.url).href,
      label: 'Reference',
      url: 'https://pan.baidu.com/s/1l6tRwttNmbf6TJNlBzIs6A?pwd=amrs'
    },
  
    {
      img: new URL('../../assets/wangluo.png', import.meta.url).href,
      label: 'Model',
      url: 'https://pan.baidu.com/s/1eRbFdVt9DXlV-N-ukttmLg?pwd=1sh9'
    }
  ])
  // let base = +new Date(1988, 9, 3);
  // let oneDay = 24 * 3600 * 1000;
  // let data = [[base, Math.random() * 300]];
  // for (let i = 1; i < 20000; i++) {
  //   let now = new Date((base += oneDay));
  //   data.push([+now, Math.round((Math.random() - 0.5) * 20 + data[i - 1][1])]);
  // }
  let titles = ['Logit', 'Prediction', 'Margin', 'LV', 'CDV', 'PV']
  let Xdata = ['logit_x', 'Prediction_x', 'Margin_x', 'LV_x', 'CDV_x', 'PV_x']
  
  let colorList = [
  '#F886A8',
  '#FE8D6F',
  '#FDC453',
  '#DFDD6C',
  '#AODDEO',
  '#9ADBC5',
  ]
  let charts = document.getElementsByClassName('main')
  
  const drawLineEchart = () => {
    for (let i = 0; i < charts.length; i++) {
      let myCharts = echarts.init(charts[i])
      myCharts.setOption({
        title: {
          left: 'center',
          text: `${titles[i]}, Epoch=${numberData}`,
          textStyle: {
            fontSize: '16',
            fontWeight: '400'
          }
        },
        toolbox: {
          feature: {
            dataZoom: {
              yAxisIndex: 'none'
            },
            restore: {},
            saveAsImage: {}
          }
        },
        xAxis: {
          type: 'category',
          boundaryGap: false,
          data: echartsData.get(Xdata[i])
        },
        yAxis: {
          type: 'value',
          boundaryGap: [0, '100%']
        },
        series: [
          {
            name: 'Fake Data',
            type: 'line',
            smooth: true,
            symbol: 'none',
            itemStyle: {
              color: colorList[i]
            },
            areaStyle: {
              color: colorList[i]
            },
            data: echartsData.get(titles[i]),
          }
        ]
      })
    }
  }
  let echartsData = new Map()
  let numberData = 0
  let clearTime
  
  const getEcharts = (range) => {
    home({ range: range }).then((res) => {
      echartsData.set('Logit', res.logit)
      echartsData.set('CDV', res.CDV)
      echartsData.set('LV', res.LV)
      echartsData.set('Margin', res.Margin)
      echartsData.set('Prediction', res.Prediction)
      echartsData.set('PV', res.PV)
      echartsData.set('logit_x', res.logit_x.map((num) => {
        return num.toFixed(2) == 0 ? 0 : num.toFixed(2) 
      }))
      echartsData.set('CDV_x', res.CDV_x.map((num) => {
        return num.toFixed(2) == 0 ? 0 : num.toFixed(2) 
      }))
      echartsData.set('LV_x', res.LV_x.map((num) => {
        return num.toFixed(2) == 0 ? 0 : num.toFixed(2) 
      }))
      echartsData.set('Margin_x', res.Margin_x.map((num) => {
        return num.toFixed(2) == 0 ? 0 : num.toFixed(2) 
      }))
      echartsData.set('Prediction_x', res.Prediction_x.map((num) => {
        return num.toFixed(2) == 0 ? 0 : num.toFixed(2) 
      }))
      echartsData.set('PV_x', res.PV_x.map((num) => {
        return num.toFixed(2) == 0 ? 0 : num.toFixed(2) 
      }))
      
    })
    drawLineEchart()
  }
  const mathNumber = () => {
    clearTime = setInterval(() => {
      if (numberData == 120) {
        numberData = 0
      } else {
        numberData++
        getEcharts(numberData)
      }
    }, 1000)
    
  }
  onMounted(() => {
    mathNumber()
  })
  onBeforeUnmount(() => {
    clearInterval(clearTime)
  })
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
  .spanLable{
    width: 250px;
    overflow: hidden;
    white-space: nowrap;
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
  
  .lineEchart {
    display: flex;
    justify-content: space-between;
    flex-wrap: wrap;
    padding: 60px 100px;
  }
  </style>