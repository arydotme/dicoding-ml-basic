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
  <div class="p-8 min-h-screen bg-gray-50 w-screen">
    <div class="max-w-3xl ml-58">

      <!-- Header -->
      <h1 class="text-3xl font-semibold text-gray-800 mb-6">Fraud Transaction Prediction</h1>

    <!-- Form Container -->
    <form
      @submit.prevent="getPrediction"
      class="bg-white p-8 rounded-xl shadow-md grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6"
    >
      <!-- Input Component -->
      <div class="flex flex-col">
        <label class="font-medium mb-1">Transaction Amount</label>
        <input v-model="transactionAmount"
               type="number"
               placeholder="Enter amount..."
               class="input-style" />
      </div>

      <div class="flex flex-col">
        <label class="font-medium mb-1">Customer Age</label>
        <input v-model="customerAge"
               type="number"
               placeholder="Enter age..."
               class="input-style" />
      </div>

      <div class="flex flex-col">
        <label class="font-medium mb-1">Transaction Duration</label>
        <input v-model="transactionDuration"
               type="number"
               placeholder="Duration..."
               class="input-style" />
      </div>

      <div class="flex flex-col">
        <label class="font-medium mb-1">Login Attempts</label>
        <input v-model="loginAttempts"
               type="number"
               placeholder="Attempts..."
               class="input-style" />
      </div>

      <div class="flex flex-col">
        <label class="font-medium mb-1">Account Balance</label>
        <input v-model="accountBalance"
               type="number"
               placeholder="Balance..."
               class="input-style" />
      </div>

      <div class="flex flex-col">
        <label class="font-medium mb-1">Location</label>
        <input v-model="location"
               placeholder="Enter location..."
               class="input-style" />
      </div>

      <div class="flex flex-col">
        <label class="font-medium mb-1">Customer Occupation</label>
        <input v-model="customerOccupation"
               placeholder="Occupation..."
               class="input-style" />
      </div>

      <div class="flex flex-col">
        <label class="font-medium mb-1">Transaction Channel</label>
        <select v-model="transactionChannel" class="input-style">
          <option disabled value="">Select Channel</option>
          <option value="ATM">ATM</option>
          <option value="Online">Online</option>
          <option value="Branch">Branch</option>
        </select>
      </div>

      <div class="flex flex-col">
        <label class="font-medium mb-1">Transaction Type</label>
        <select v-model="transactionType" class="input-style">
          <option disabled value="">Select Type</option>
          <option value="Debit">Debit</option>
          <option value="Credit">Credit</option>
        </select>
      </div>

      <!-- Submit Button -->
      <div class="col-span-full flex justify-start">
        <button
          class="px-6 py-3 bg-blue-600 text-white rounded-lg font-semibold hover:bg-blue-700 transition"
        >
          Get Prediction
        </button>
      </div>

      <!-- Result -->
      <div
        v-if="predictionResult !== null"
        class="col-span-full bg-green-100 text-green-800 p-4 rounded-lg font-semibold"
      >
        Prediction : {{ predictionResult }}
      </div>
    </form>
    </div>
  </div>
</template>

<style scoped>

</style>