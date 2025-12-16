<script setup>
import {ref, onMounted, callWithAsyncErrorHandling} from "vue";
  import axios from "axios";
  import Chart from "chart.js/auto";

  const chart = ref(null);

  onMounted(async () => {
    const { data } = await axios.post("http://localhost:8000/features-important");

    new Chart(chart.value, {
      type: "bar",
      data: {
        labels: data.feature,
        datasets: [
          {
            label: "Mean Absolute Contribution",
            data: data.values,
          },
        ],
      },
    });
  });
</script>

<template>
  <div class="bg-white p-6 rounded shadow w-200">
    <h2 class="font-bold text-lg mb-4">Feature Importance</h2>
    <canvas ref="chart"></canvas>
  </div>
</template>

<style scoped>

</style>