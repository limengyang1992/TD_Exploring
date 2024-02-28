<template>
    <div class="content">
        <div class="title">
            <p>Application: Imbalance Learning</p>
            <div class="des">
                <div>Many meta-learning-based imbalance learning algorithms, such as Meta-Weight-Net, employ limited raw quantities to infer the weights of samples. We replace the raw TD quantities with our deep representations. Specifically, the TD quantities are extracted during training and input into TD2Vec to generate deep representations of samples. Then, the representations are input into the weighting network to generate the weights for the next-epoch training. Moreover, a meta-learning-based algorithm that also adopts the bi-level optimization scheme is utilized to update the backbone and the weighting networks. Our method differs from existing methods in two folds. First, a wide range of quantities is considered. Second, the deep representation characterizing the training dynamics from the first to the current epochs is used, whereas previous methods employ the quantities in the current epoch. The upgraded version of the Meta-Weight-Net algorithm, which considers both deep representation and temporal sequence relationships, is called MWN-DRTS, as shown in the following figure.</div>
            </div>
        </div>
    </div>
    <div class="gnld">
        <!-- <div class="title">Noise detection</div> -->
        <img src="../../assets/learning.png" alt="">
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
                [10, 81.64, 'CE Loss'],
                [23, 81.89, 'Focal loss'],
                [28, 82.82, 'CB'],
                [35, 83.79, 'CB Focal loss'],
                [40, 80.66, 'CB fine-tuning'],
                [46, 80.87, 'L2RW'],
                [52, 83.66, 'LDAM'],
                [58, 85.79, 'LDAM-DRW '],
                [64, 86.18, 'LPL'],
                [69, 85.80, 'IB loss'],
                [75, 83.72, 'Meta-Weight-Net'],
                [83, 85.44, 'Meta-Class-Weight'],
                [86, 86.98, 'MWN-TDQ'],
                [94, 87.79, 'MWN-DRTS'],
            ]
        },
        // grid: { containLabel: true },
        xAxis: { name: '' },
        yAxis: { type: 'category' },
        visualMap: {
            orient: 'horizontal',
            left: 'center',
            min: 0,
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