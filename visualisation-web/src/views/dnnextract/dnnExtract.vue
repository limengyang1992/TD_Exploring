<template>
    <div class="extract">
        <el-table :data="tableData" style="width: 100%" border>
            <el-table-column fixed prop="Quantity" label="Quantity" width="100">
                <template #default="scope">
                    <vue-latex :expression="scope.row.Quantity" display-mode />
                </template>
            </el-table-column>
            <el-table-column prop="Formula" label="Formula" width="300">
                <template #default="scope">
                    <vue-latex :expression="scope.row.Formula" display-mode />
                </template>
            </el-table-column>
            <el-table-column prop="Description" label="Description">
            </el-table-column>
        </el-table>
        
    </div>
</template>
<script setup>
import { ref } from 'vue'
const tableData = ref([
    {
        "Quantity": "\\zeta_{i,48}^{t}",
        "Formula": "{r(o_{i}^{t})}/{r(o_{i}^{t-1})}",
        "Description": " The variation of the normalized rank of the logit in the corpus from the $(t-1)$-th to $t$-th epoch "
    },
    {
        "Quantity": "\\zeta_{i,104}^{t}",
        "Formula": "r(cos(o_{i}^{t},\\bar{o}_{\\mathcal{N}_{i,k}^t}))/r(cos(o_{i}^{t-1},\\bar{o}_{\\mathcal{N}_{i,k}^{t-1}}))",
        "Description": " The variation of the normalized rank of the cosine distance between the sample logit and the average neighborhood logit in the corpus from the $(t-1)$-th to $t$-th epoch "
    },
    {
        "Quantity": "\\zeta_{i,36}^{t}",
        "Formula": "o_{i}^{t}/o_{i}^{t-1}",
        "Description": " The variation of the normalized logit in the corpus from the $(t-1)$-th to $t$-th epoch "
    },
    {
        "Quantity": "\\zeta_{i,96}^{t}",
        "Formula": "r(p_{i}^{t})/r(p_{i}^{t-1})",
        "Description": " The variation of the normalized rank of the prediction in the corpus from the $(t-1)$-th to $t$-th epoch "
    },
    {
        "Quantity": "\\zeta_{i,84}^{t}",
        "Formula": "p_{i}^{t}/p_{i}^{t-1}",
        "Description": " The variation of the normalized prediction in the corpus from the $(t-1)$-th to $t$-th epoch "
    },
    {
        "Quantity": "\\zeta_{i,128}^{t}",
        "Formula": "r(\\gamma_{i}^{t})",
        "Description": " The normalized rank of the margin in the corpus"
    },
    {
        "Quantity": "\\zeta_{i,76}^{t}",
        "Formula": "{r({\\gamma}_{i}^{t})}/{r(\\gamma_{i}^{t-1})}",
        "Description": " The variation of the normalized rank of the margin in the corpus from the $(t-1)$-th to $t$-th epoch "
    },
    {
        "Quantity": "\\zeta_{i,97}^{t}",
        "Formula": "r(cos(o_{i}^{t},\\bar{o}_{\\mathcal{N}_{i,k}^t}))/r(cos(o_{i}^{t-1},\\bar{o}_{\\mathcal{N}_{i,k}^{t-1}}))",
        "Description": " The variation of the normalized rank of the cosine distance between the sample logit and the average neighborhood logit in the corpus from the $(t-1)$-th to $t$-th epoch "
    },
    {
        "Quantity": "\\zeta_{i,136}^{t}",
        "Formula": "\\gamma_{i}^{t}/\\gamma_{i}^{t-1}",
        "Description": " The variation of the normalized margin in the corpus from the $(t-1)$-th to $t$-th epoch "
    },
    {
        "Quantity": "\\zeta_{i,16}^{t}",
        "Formula": "r(l_{i}^{t})/r(l_{i}^{t-1})",
        "Description": " The variation of the normalized rank of the loss in the corpus from the $(t-1)$-th to $t$-th epoch "
    },
    {
        "Quantity": "\\zeta_{i,72}^{t}",
        "Formula": "r(g_{i}^{t})/r(g_{i}^{t-1})",
        "Description": " The variation of the normalized rank of the gradient in the corpus from the $(t-1)$-th to $t$-th epoch "
    },
    {
        "Quantity": "\\zeta_{i,112}^{t}",
        "Formula": "r(e_{i}^{t})/r(e_{i}^{t-1})",
        "Description": " The variation of the normalized rank of the entropy in the corpus from the $(t-1)$-th to $t$-th epoch "
    },
    {
        "Quantity": "\\zeta_{i,120}^{t}",
        "Formula": "e_{i}^{t}/e_{i}^{t-1}",
        "Description": " The variation of the normalized entropy in the corpus from the $(t-1)$-th to $t$-th epoch "
    },
    {
        "Quantity": "\\zeta_{i,140}^{t}",
        "Formula": "(\\bar{\\gamma}_{\\mathcal{N}_{i,k}^{t}}/\\bar{\\gamma}_{y_{i}}^{t})/(\\bar{\\gamma}_{\\mathcal{N}_{i,k}^{t-1}}/\\bar{\\gamma}_{y_{i}}^{t-1})",
        "Description": " The variation of the ratio between the average neighborhood margin and the average category margin from the $(t-1)$-th to $t$-th epoch "
    },
    {
        "Quantity": "\\zeta_{i,68}^{t}",
        "Formula": "(g_{i}^{t}/\\bar{g}_{y_i}^{t})/(g_{i}^{t-1}/\\bar{g}_{y_i}^{t-1})",
        "Description": " The variation of the ratio between the sample gradient and average category gradient from the $(t-1)$-th to $t$-th epoch "
    },
    {
        "Quantity": "\\zeta_{i,90}^{t}",
        "Formula": "r(cos(o_{i}^{t},\\bar{o}_{\\mathcal{N}_{i,k}^{t}}))",
        "Description": " The normalized rank of the cosine distance between the sample logit and the average neighborhood logit in the corpus"
    },
    {
        "Quantity": "\\zeta_{i,64}^{t}",
        "Formula": "r(g_{i}^{t})/r(g_{i}^{t-1})",
        "Description": " The variation of the normalized rank of the gradient in the corpus from the $(t-1)$-th to $t$-th epoch "
    },
    {
        "Quantity": "\\zeta_{i,94}^{t}",
        "Formula": "r(cos(o_{i}^{t},\\bar{o}_{y_{i}}^{t}))/r(cos(o_{i}^{t-1},\\bar{o}_{y_{i}}^{t-1}))",
        "Description": " The variation of the normalized rank of the cosine distance between the sample logit and the average category logit from the $(t-1)$-th to $t$-th epoch "
    },
    {
        "Quantity": "\\zeta_{i,106}^{t}",
        "Formula": "(e_{i}^{t}/\\bar{e}_{y_i}^{t})/(e_{i}^{t-1}/\\bar{e}_{y_i}^{t-1})",
        "Description": "The variation of the ratio between the sample entropy and the average category entropy from the $(t-1)$-th to $t$-th epoch "
    },
    {
        "Quantity": "\\zeta_{i,110}^{t}",
        "Formula": "(e_{i}^{t}/\\bar{e}_{C{p}}^{t})/(e_{i}^{t-1}/\\bar{e}_{C{p}}^{t-1})",
        "Description": "The variation of the ratio between the sample entropy and the average corpus entropy from the $(t-1)$-th to $t$-th epoch "
    }
])
</script>
<style scoped>
.extract {
    width: 1400px;
    margin: 60px auto;
}
</style>