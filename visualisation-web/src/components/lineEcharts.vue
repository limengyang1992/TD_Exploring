<template>
    <div id="main"></div>
</template>
<script setup>
import { onMounted, computed } from 'vue'
import * as echarts from 'echarts';
const props = defineProps({
    lineEchartData: {
        type: Array,
        default: () => {
      return []
    }
    }
})
const lineEchartDatas = computed(() => {
    return props.lineEchartData
})

var option;

option = {
    backgroundColor: '#fff',
    xAxis: {
        type: 'category',
        data: ['0', '20', '40', '60', '80', '100', '120']
    },
    yAxis: {
        type: 'value'
    },
    series: [
        {
            data: lineEchartDatas.value,
            type: 'line',
            smooth: true
        }
    ],
    grid: {
        left: '1%', // 与容器左侧的距离
        right: '2%', // 与容器右侧的距离
        top: '10%', // 与容器顶部的距离
        bottom: '10%', // 与容器底部的距离
        borderWidth: 10,
        containLabel: true,
    },
    // dataZoom: [
    //     {
    //         show: true,
    //         height: 8,
    //         xAxisIndex: [0],
    //         bottom: "6%",
    //         start: 0,
    //         end: 50,
    //         handleIcon:
    //             "path://M306.1,413c0,2.2-1.8,4-4,4h-59.8c-2.2,0-4-1.8-4-4V200.8c0-2.2,1.8-4,4-4h59.8c2.2,0,4,1.8,4,4V413z",
    //         handleSize: "110%",
    //         handleStyle: {
    //             color: "#d3dee5",
    //         },
    //         textStyle: {
    //             color: "#fff",
    //         },
    //         borderColor: "#90979c",
    //     },
    //     {
    //         type: "inside",
    //         show: true,
    //         height: 15,
    //         start: 1,
    //         end: 35,
    //     },
    // ],
};
onMounted(() => {
    var myCharts = echarts.init(document.getElementById('main'), null, { renderer: 'svg' });
    myCharts.clear();
    myCharts.setOption(option)
    // 当窗口或者大小发生改变时执行resize，重新绘制图表
    window.addEventListener('resize', function () {
        myCharts.resize()
    })
})
</script>
<style scoped>
#main {
    max-width: 400px;
    height: 300px;
}
</style>