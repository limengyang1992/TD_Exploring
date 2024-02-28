<template>
  <div class="back_color">
    <div class="about">
      <div class="filter">
        <el-form :model="formData" label-width="120px">
          <el-row :gutter="20">
            <el-col :span="24" v-if="formData.ModelData.ModelList">
              <el-form-item label="Model">
                <el-radio-group v-model="formData.ModelData.ModelLable" text-color="#F8466F">
                  <template v-for="item, index in formData.ModelData.ModelList" :key="index">
                    <el-radio-button :label="item.label">{{ item.label }}</el-radio-button>
                  </template>
                </el-radio-group>
              </el-form-item>
            </el-col>
            <el-col :span="24" v-if="formData.DatasetData.DatasetList">
              <el-form-item label="Dataset">
                <el-radio-group v-model="formData.DatasetData.DatasetLable" text-color="#F8466F">
                  <template v-for="item, index in formData.DatasetData.DatasetList" :key="index">
                    <el-radio-button :label="item.label">{{ item.label }}</el-radio-button>
                  </template>
                </el-radio-group>
              </el-form-item>
            </el-col>
            <el-col :span="24" v-if="formData.ClassData.ClassList">
              <el-form-item label="Class">
                <el-radio-group v-model="formData.ClassData.ClassLable" text-color="#F8466F">
                  <template v-for="item, index in formData.ClassData.ClassList" :key="index">
                    <el-radio-button :label="item.label">{{ item.label }}</el-radio-button>
                  </template>
                </el-radio-group>
              </el-form-item>
            </el-col>
            <el-col :span="24" v-if="formData.IDData.IDList">
              <el-form-item label="ID">
                <el-radio-group v-model="formData.IDData.IDLable" text-color="#F8466F">
                  <template v-for="item, index in formData.IDData.IDList" :key="index">
                    <el-radio-button :label="item.value">{{ item.label }}</el-radio-button>
                  </template>
                </el-radio-group>
              </el-form-item>
            </el-col>
            <el-col :span="24" v-if="formData.epochData.epochList">
              <el-form-item label="Epoch">
                <el-radio-group v-model="formData.epochData.epochLable" text-color="#F8466F">
                  <template v-for="item, index in formData.epochData.epochList" :key="index">
                    <el-radio-button :label="item.value">{{ item.label }}</el-radio-button>
                  </template>
                </el-radio-group>
              </el-form-item>
            </el-col>
            <!-- <el-col :span="24" v-if="formData.TDqualitiesData.TDqualitiesList"> -->
            <el-col :span="24" v-if="false">
              <el-form-item label="TD">
                <el-radio-group v-model="formData.TDqualitiesData.TDqualitiesLable" text-color="#F8466F">
                  <template v-for="item, index in formData.TDqualitiesData.TDqualitiesList" :key="index">
                    <el-radio-button :label="item.label">{{ item.label }}</el-radio-button>
                  </template>
                </el-radio-group>
              </el-form-item>
            </el-col>
          </el-row>
        </el-form>
      </div>
      <hr>
      <div class="imgList">
        <template v-for="item in imgData" :key="item">
          <!-- <img :src="item" alt=""> -->
          <!-- <img :src="`data:image/png;base64,${item}`" alt=""> -->
        </template>
      </div>
      
      <hr>
      <div class="lineEchart">
        <template v-for="item, index in lineEchartData" :key="item">
          <div class="box">
            <div class="main" style="width: 200px;height: 200px;"></div>
          <p>{{ nameData[index] }}</p>
          </div>
        </template>
      </div>
      <!-- <hr>
      <div class="lineEchart">
          <div id="barmain" style="width: 100%;height: 300px;"></div>
      </div> -->
    </div>
  </div>
</template>

<script setup>
import { ref, watch } from 'vue';
// import lineEcharts from '../../components/lineEcharts.vue'
// import barEcharts from '../../components/barEcharts.vue'
import { queryperception } from '../../services/data'
import * as echarts from 'echarts';
const formData = ref({
  ModelData: {
    ModelLable: 'Vgg16',
    ModelList: [
      {
        value: 'Vgg16',
        label: 'Vgg16'
      },
      {
        value: 'ResNet18',
        label: 'ResNet18'
      },
      {
        value: 'ResNet34',
        label: 'ResNet34'
      },
      {
        value: 'DenseNet',
        label: 'DenseNet'
      }
    ]
  },
  DatasetData: {
    DatasetLable: 'Random',
    DatasetList: [
    {
        value: 'Random',
        label: 'Random'
      },
      {
        value: 'CIFAR100',
        label: 'CIFAR100'
      },
      {
        value: 'MNIST',
        label: 'MNIST'
      },
      {
        value: 'Flowers102',
        label: 'Flowers102'
      },
      {
        value: 'ImageNet-LT',
        label: 'ImageNet-LT'
      }
    ]
  },
  ClassData: {
    ClassLable: 'Random',
    ClassList: [
      {
        value: 'Random',
        label: 'Random'
      },
      {
        value: 'clean',
        label: 'Clean'
      },
      {
        value: 'noise',
        label: 'Noise'
      },
      {
        value: 'head',
        label: 'Head'
      },
      {
        value: 'tail',
        label: 'Tail'
      },
      {
        value: 'adver',
        label: 'Adver'
      }
    ]
  },
  IDData: {
    IDLable: 'Random',
    IDList: [
      {
        value: 'Random',
        label: 'Random'
      },
      {
        value: '1',
        label: 'No.1'
      },
      {
        value: '2',
        label: 'No.2'
      },
      {
        value: '3',
        label: 'No.3'
      },
      {
        value: '4',
        label: 'No.4'
      },
      {
        value: '5',
        label: 'No.5'
      }
    ]
  },
  epochData: {
    epochLable: 'Random',
    epochList: [
      {
        value: 'Random',
        label: 'Random'
      },
      {
        value: '0',
        label: 'Epoch.0'
      },
      {
        value: '30',
        label: 'Epoch.30'
      },
      {
        value: '60',
        label: 'Epoch.60'
      },
      {
        value: '90',
        label: 'Epoch.90'
      },
      {
        value: '120',
        label: 'Epoch.120'
      }
    ]
  },
  TDqualitiesData: {
    TDqualitiesLable: 'Random',
    TDqualitiesList: [
      {
        value: 'Random',
        label: 'Random'
      },
      {
        value: 'Loss',
        label: 'Loss'
      },
      {
        value: 'Logit',
        label: 'Logit'
      },
      {
        value: 'Prediction',
        label: 'Prediction'
      },
      {
        value: 'Margin',
        label: 'Margin'
      },
      {
        value: 'Gradient',
        label: 'Gradient'
      },
      {
        value: 'Uncertainty',
        label: 'Uncertainty'
      },
      {
        value: 'Forgotten',
        label: 'Forgotten'
      }
    ]
  }
})
const options = ref({
  model: '',
  dataset: '',
  classes: '',
  id: '',
  td: '',
  epoch: ''
})
const drawLineEchart = () => {
  let charts = document.getElementsByClassName('main')
  for (let i = 0; i < charts.length; i++) {
    let myCharts = echarts.init(charts[i])
    myCharts.setOption({
      backgroundColor: '#fff',
      xAxis: {
        type: 'category',
        data: ''
        
      },
      yAxis: {
        type: 'value'
      },
      series: [
        {
          data: lineEchartData.value[i],
          type: 'line',
          smooth: true,
          symbol: 'none',
          lineStyle: { // X 轴颜色配置
              color: '#9ADBC5'  
            } 
        }
      ],
      grid: {
        left: '1%', // 与容器左侧的距离
        right: '2%', // 与容器右侧的距离
        top: '10%', // 与容器顶部的距离
        bottom: '10%', // 与容器底部的距离
        borderWidth: 10,
        containLabel: true,
      }
    })
  }
}
const drawBarEchart = () => {
  let charts = document.getElementById('barmain')
  let myCharts = echarts.init(charts)
    myCharts.setOption({
      xAxis: {
        data: '',
        axisLabel: {
          inside: true,
          color: '#fff'
        },
        axisTick: {
          show: false
        },
        axisLine: {
          show: false
        },
        z: 10
      },
      yAxis: {
        axisLine: {
          show: false
        },
        axisTick: {
          show: false
        },
        axisLabel: {
          color: '#999'
        }
      },
      // dataZoom: [
      //   {
      //     type: 'inside'
      //   }
      // ],
      series: [
        {
          type: 'bar',
          showBackground: true,
          itemStyle: {
            color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
              { offset: 0, color: '#83bff6' },
              { offset: 0.5, color: '#188df0' },
              { offset: 1, color: '#188df0' }
            ]),
          },
          emphasis: {
            itemStyle: {
              color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
                { offset: 0, color: '#2378f7' },
                { offset: 0.7, color: '#2378f7' },
                { offset: 1, color: '#83bff6' }
              ])
            }
          },
          data: barEchartsData.value
        }
      ]
    })
}
const lineEchartData = ref([])
const barEchartsData = ref([])
const imgData = ref([])
const nameData = ref([])
const onSubmit = () => {
  options.value = {
    ...options.value,
    model: formData.value.ModelData.ModelLable,
    dataset: formData.value.DatasetData.DatasetLable == 'Random' ? 'random' : formData.value.DatasetData.DatasetLable,
    classes: formData.value.ClassData.ClassLable == 'Random' ? 'random' : formData.value.ClassData.ClassLable,
    id: formData.value.IDData.IDLable == 'Random' ? 'random' : formData.value.IDData.IDLable,
    td: formData.value.TDqualitiesData.TDqualitiesLable == 'Random' ? 'random' : formData.value.TDqualitiesData.TDqualitiesLable,
    epoch: formData.value.epochData.epochLable == 'Random' ? 'random' : formData.value.epochData.epochLable
  }
  queryperception(options.value).then((res) => {
    console.log(res, 123)
    lineEchartData.value = res.vectors_120d
    barEchartsData.value = res.vector_100d
    imgData.value = res.images
    nameData.value = res.names_120d

  })
  setTimeout(() => {
    drawLineEchart()
    drawBarEchart()
  }, 2000);
}
onSubmit()
watch(
  () => formData.value,
  () => {
    onSubmit()
  },
  {deep: true}
)
</script>

<style scoped>
.about {
  width: 1400px;
  margin: 0 auto;
  padding-top: 40px;
}

.filter {
  padding: 0 60px;
}

:deep(.el-radio) {
  width: 70px;
}

:deep(.el-radio__input.is-checked .el-radio__inner) {
  border-color: #F8466F;
  background-color: #F8466F;
}

:deep(.el-radio__input.is-checked+.el-radio__label) {
  color: #F8466F;
}

:deep(.el-form-item__label) {
  font-weight: 500;
  font-size: 20px;
}
:deep(.el-form-item) {
  margin-bottom: 0;
}

.submit {
  display: flex;
  justify-content: flex-end;
}

:deep(.el-button--primary) {
  background-color: #F8466F;
  border: 1px solid #F8466F;
}

/* .back_color {
  background-color: #F3F4FA;
} */

hr {
  display: inline-block;
  text-align: center;
  width: 100%;
  height: 1px;
  background-color: #E8ECF4;
  color: #cecece;
  margin: 24px 0;
  border: 0px;
}

.lineEchart {
  padding: 30px 80px;
  display: flex;
  flex-wrap: wrap;
  justify-content: space-between;
}

:deep(.el-radio-button__inner) {
  background: none !important;
  border: none;
  border-left: none;
  color: #6F6F6F;
  width: 120px;
  text-align: left;
  font-size: 20px;
  font-weight: 200;
}

:deep(.el-radio-button:first-child .el-radio-button__inner),
:deep(.el-radio-button__original-radio:checked+.el-radio-button__inner) {
  border: none;
  box-shadow: none;
}

.imgList {
  display: flex;
  justify-content: space-between;
  align-items: end;
  padding: 0 80px;
}
.imgList > img {
  width: 200px;
  height: 200px;
}
.imgList > img:not(:first-child) {
  height: 160px;
}
.main {
  padding-right: 5px ;
  position: relative;
}
.box {
  display: flex;
  flex-direction: column;
  align-items: center;
}
.box > p {
  padding-bottom: 30px;
}
</style>
