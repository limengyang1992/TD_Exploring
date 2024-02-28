<template>
    <div class="contrast">
        <div class="left">
            <form :model="formData" label-width="100px">
                <el-row :gutter="20">
                    <el-col :span="10">
                        <el-form-item label="任务数据集">
                            <el-select v-model="formData.dataList" placeholder="please select your zone">
                                <el-option label="Zone one" value="shanghai" />
                                <el-option label="Zone two" value="beijing" />
                            </el-select>
                        </el-form-item>
                    </el-col>
                    <el-col :span="10">
                        <el-form-item label="任务模型">
                            <el-select v-model="formData.dataModel" placeholder="please select your zone">
                                <el-option label="Zone one" value="shanghai" />
                                <el-option label="Zone two" value="beijing" />
                            </el-select>
                        </el-form-item>
                    </el-col>
                    <el-col :span="10">
                        <el-form-item label="数据类别">
                            <el-select v-model="formData.dataType" placeholder="please select your zone">
                                <el-option label="Zone one" value="shanghai" />
                                <el-option label="Zone two" value="beijing" />
                            </el-select>
                        </el-form-item>
                    </el-col>
                    <el-col :span="10">
                        <el-form-item label="数据样本">
                            <el-select v-model="formData.dataSample" placeholder="please select your zone">
                                <el-option label="Zone one" value="shanghai" />
                                <el-option label="Zone two" value="beijing" />
                            </el-select>
                        </el-form-item>
                    </el-col>
                    <el-col :span="10">
                        <el-form-item label="TD">
                            <el-select v-model="formData.dataTd" placeholder="please select your zone">
                                <el-option label="Zone one" value="shanghai" />
                                <el-option label="Zone two" value="beijing" />
                            </el-select>
                        </el-form-item>
                    </el-col>
                    <el-col :span="10">
                        <el-form-item>
                            <el-button type="primary" @click="onSubmit">搜索</el-button>
                        </el-form-item>
                    </el-col>
                </el-row>
            </form>
        </div>
        <div class="right">
            <form :model="formData" label-width="100px">
                <el-row :gutter="20">
                    <el-col :span="10">
                        <el-form-item label="任务数据集">
                            <el-select v-model="formData.dataList" placeholder="please select your zone">
                                <el-option label="Zone one" value="shanghai" />
                                <el-option label="Zone two" value="beijing" />
                            </el-select>
                        </el-form-item>
                    </el-col>
                    <el-col :span="10">
                        <el-form-item label="任务模型">
                            <el-select v-model="formData.dataModel" placeholder="please select your zone">
                                <el-option label="Zone one" value="shanghai" />
                                <el-option label="Zone two" value="beijing" />
                            </el-select>
                        </el-form-item>
                    </el-col>
                    <el-col :span="10">
                        <el-form-item label="数据类别">
                            <el-select v-model="formData.dataType" placeholder="please select your zone">
                                <el-option label="Zone one" value="shanghai" />
                                <el-option label="Zone two" value="beijing" />
                            </el-select>
                        </el-form-item>
                    </el-col>
                    <el-col :span="10">
                        <el-form-item label="数据样本">
                            <el-select v-model="formData.dataSample" placeholder="please select your zone">
                                <el-option label="Zone one" value="shanghai" />
                                <el-option label="Zone two" value="beijing" />
                            </el-select>
                        </el-form-item>
                    </el-col>
                    <el-col :span="10">
                        <el-form-item label="TD">
                            <el-select v-model="formData.dataTd" placeholder="please select your zone">
                                <el-option label="Zone one" value="shanghai" />
                                <el-option label="Zone two" value="beijing" />
                            </el-select>
                        </el-form-item>
                    </el-col>
                    <el-col :span="10">
                        <el-form-item>
                            <el-button type="primary" @click="onSubmit">搜索</el-button>
                        </el-form-item>
                    </el-col>
                </el-row>
            </form>
        </div>
    </div>
</template>
<script setup>
import { ref } from 'vue';
const formData = ref({
  dataList: [],
  dataModel: [],
  dataType: [],
  dataSample: [],
  dataTd: []
})
</script>
<style scoped>
.contrast {
    display: flex;
}

.contrast>div {
    flex: 1;
    padding: 0 20px;
}
.left {
    border-right: 4px solid;
  border-image: linear-gradient(to right, #8f41e9, #578aef) 1;
}
:deep(.el-form-item.asterisk-left) {
    display: flex;
    align-items: center;
    margin-top: 20px;
}
:deep(.el-form-item__content) {
    flex: 2;
    margin-left: 20px;
}
:deep(.el-form-item__label) {
    flex: 1;
    text-align: right;
}
</style>