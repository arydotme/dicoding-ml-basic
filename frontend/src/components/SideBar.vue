<script setup>
</script>

<template>
  <aside class="fixed left-0 top-0 h-full w-56 bg-white shadow-lg p-5 flex flex-col">
    <h2 class="text-xl font-bold text-gray-800 mb-6">Menu</h2>

    <nav class="flex flex-col gap-2">
      <router-link
        to="/dashboard"
        class="flex items-center gap-3 p-3 rounded-lg text-gray-700 hover:bg-gray-100 transition"
      >
        <span>📊</span> Dashboard
      </router-link>

      <router-link
        to="/form"
        class="flex items-center gap-3 p-3 rounded-lg text-gray-700 hover:bg-gray-100 transition"
      >
        <span>📝</span> Get Predict
      </router-link>
    </nav>
  </aside>
</template>

<style scoped>

</style>