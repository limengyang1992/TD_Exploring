<template>
    <div class="content">
        <div class="title">
            <p>Application: Noise Detection </p>
            <div class="des">
                <div>Many noisy label learning methods first detect noisy labels and then train a model with the detected results.  Compared with noisy label learning, we believe that there exists a more intricate relationship between exploring training dynamics and noisy label detection tasks. Noisy label detection is actually a binary classification task. However, to our knowledge, there has been no study that aims to construct a general noisy label detector with a standard supervised learning paradigm. Our learning framework for TD representations offers a feasible way to train a general noisy label detector (GNLD), as depicted in the following figure. </div>
            </div>
        </div>
    </div>
    <div class="gnld">
        <!-- <div class="title">Imbalance learning</div> -->
        <img src="../../assets/gnld.png" alt="">
        <div class="des">
             <div> </div> 
        </div>
    </div>
    <div class="lineEchart">
        <div id="main" style="width: 1200PX;height: 0.1px;"></div>
    </div>
</template>
<script setup>
import { onMounted } from 'vue';
import * as echarts from 'echarts';
const drawBarEchart = () => {
    let charts = document.getElementById('main')
    let myCharts = echarts.init(charts)
    myCharts.setOption({
        dataset: {
            source: [
                ['score', 'amount', 'product'],
                [10, 53.56, 'Loss'],
                [15, 62.67, 'MentorNet'],
                [20, 72.59, 'Co-teaching'],
                [25, 81.70, 'O2UNet'],
                [30, 79.62, 'VOG'],
                [37, 79.26, 'SimiFeat'],
                [48, 83.06, 'SPR'],
                [50, 87.30, 'GNLD'],
                [60, 88.50, 'GNLD+DA'],
                [70, 88.94, 'GNLD+FT'],
                [89, 91.28, 'GNLD+DA+FT'],
                // [50, 91, 'C'],
                // [50, 10, 'V'],
                // [50, 20, 'B']
            ]
        },
        // grid: { containLabel: true },
        xAxis: { name: '' },
        yAxis: { type: 'category' },
        visualMap: {
            orient: 'horizontal',
            left: 'center',
            min: 10,
            max: 100,
            show: false,
            dimension: 0,
            inRange: {
                color: ['#65B581', '#FFCE34', '#FD665F']
            }
        },
        series: [
            {
                type: 'bar',
                encode: {
                    // Map the "amount" column to X axis.
                    x: 'amount',
                    // Map the "product" column to Y axis
                    y: 'product'
                }
            }
        ]
    })
}
onMounted(() => {
    drawBarEchart()
})

</script>
<style scoped>
.gnld,
.content {
    width: 1400px;
    padding: 60px 140px;
    display: flex;
    justify-content: center;
    align-items: center;
    flex-direction: column;
    margin: 0 auto;
}
.gnld {
    padding: 60px 140px 0;
}

.content .title,
.content {
    padding-bottom: 0;
}

.content .title p {
    font-size: 50px;
    font-weight: 800;
    color: #081642;
    line-height: 1.2em;
    padding-bottom: 60px;
}

.title {
    font-size: 30px;
    font-weight: 700;
    padding-bottom: 60px;
    text-align: center;
}

.des>div {
    font-size: 20px;
    text-align: justify;
    flex: 1;
    display: flex;
    justify-content: center;
    align-items: center;
}

img {
    width: 1000px;
}

.gnld .des>div {
    text-align: center;
    width: 1000px;
    padding-top: 30px;
}
.lineEchart {
    padding: 30px 100px;
    display: flex;
    justify-content: center;
}
</style>