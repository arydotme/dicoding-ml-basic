<script setup>

import { ref } from 'vue';
import axios from "axios"

const transactionAmount = ref("");
const customerAge = ref("");
const transactionDuration = ref("");
const loginAttempts = ref("");
const accountBalance = ref("");
const location = ref("");
const customerOccupation = ref("");
const transactionChannel = ref("");
const transactionType = ref("");

const predictionResult = ref(null);

async function getPrediction() {
  const payload = {
    transaction_amount: Number(transactionAmount.value),
    customer_age: Number(customerAge.value),
    transaction_duration: Number(transactionDuration.value),
    login_attempts: Number(loginAttempts.value),
    account_balance: Number(accountBalance.value),
    location: String(location.value),
    customer_occupation: String(customerOccupation.value),
    transaction_channel: String(transactionChannel.value),
    transaction_type: String(transactionType.value),
  };

  try {
    const res = await axios.post("http://localhost:8000/predict", payload);
    predictionResult.value = res.data.prediction;
  } catch (err) {
    predictionResult.value = `Request failed: ${err?.message ?? err}`;
  }
}

</script>

<template>
  <form
      @submit.prevent="getPrediction"
      class="grid grid-cols-2 lg:grid-cols-3 w-full justify-center items-center px-70 gap-10">
    <div class="flex flex-col">
      <label>
        Transaction Amount
      </label>
      <input v-model="transactionAmount" placeholder="Type here..." class="w-fit border rounded-md border-slate-200 px-3 py-2 placeholder:text-slate-400 text-slate-700 focus:outline-none focus:border-slate-400">
    </div>
    <div class="flex flex-col">
      <label>
        Customer Age
      </label>
      <input v-model="customerAge" placeholder="Type here..." class="w-fit border rounded-md border-slate-200 px-3 py-2 placeholder:text-slate-400 text-slate-700 focus:outline-none focus:border-slate-400">
    </div>
    <div class="flex flex-col">
      <label>
        Transaction Duration
      </label>
      <input v-model="transactionDuration" placeholder="Type here..." class="w-fit border rounded-md border-slate-200 px-3 py-2 placeholder:text-slate-400 text-slate-700 focus:outline-none focus:border-slate-400">
    </div>
    <div class="flex flex-col">
      <label>
        Login Attempts
      </label>
      <input v-model="loginAttempts" placeholder="Type here..." class="w-fit border rounded-md border-slate-200 px-3 py-2 placeholder:text-slate-400 text-slate-700 focus:outline-none focus:border-slate-400">
    </div>
    <div class="flex flex-col">
      <label>
        Account Balance
      </label>
      <input v-model="accountBalance" placeholder="Type here..." class="w-fit border rounded-md border-slate-200 px-3 py-2 placeholder:text-slate-400 text-slate-700 focus:outline-none focus:border-slate-400">
    </div>
    <div class="flex flex-col">
      <label>
        Location
      </label>
      <input v-model="location" placeholder="Type here..." class="w-fit border rounded-md border-slate-200 px-3 py-2 placeholder:text-slate-400 text-slate-700 focus:outline-none focus:border-slate-400">
    </div>
    <div class="flex flex-col">
      <label>
        CustomerOccupation
      </label>
      <input v-model="customerOccupation" placeholder="Type here..." class="w-fit border rounded-md border-slate-200 px-3 py-2 placeholder:text-slate-400 text-slate-700 focus:outline-none focus:border-slate-400">
    </div>
    <div class="flex flex-col">
      <label>Transaction Channel</label>
      <select
          v-model="transactionChannel"
          class="w-full border rounded-md border-slate-200 px-3 py-2 bg-white text-slate-700 focus:outline-none focus:border-slate-400"
      >
        <option disabled value="">Transaction Channel</option>
        <option value="Atm">ATM</option>
        <option value="Online">Online</option>
        <option value="Branch">Branch</option>
      </select>
    </div>
    <div class="flex flex-col">
      <label>Transaction Type</label>
      <select
          v-model="transactionType"
          class="w-full border rounded-md border-slate-200 px-3 py-2 bg-white text-slate-700 focus:outline-none focus:border-slate-400"
      >
        <option disabled value="">Transaction Type</option>
        <option value="Debit">Debit</option>
        <option value="Credit">Credit</option>
      </select>
    </div>
    <div>
      <button class="bg-amber-700 py-2 px-4 rounded-xl">
        Get Predict
      </button>
    </div>
    <div v-if="predictionResult" class="mt-4 p-4 bg-green-100 rounded-xl">
      <p class="font-bold text-green-700">Prediction: {{ predictionResult }}</p>
    </div>
  </form>
</template>

<style scoped>

</style>