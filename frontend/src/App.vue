<script setup>
import SideBar from "./components/SideBar.vue";
</script>

<template>
  <div class="flex h-screen">
    <SideBar/>
    <div class="">
      <router-view/>
    </div>
  </div>
</template>

<style scoped></style>