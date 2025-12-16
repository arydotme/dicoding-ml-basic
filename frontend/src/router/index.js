import { createMemoryHistory ,createRouter } from "vue-router";

import Dashboard from "../pages/Dashboard.vue";
import FormInput from "../components/FormInput.vue";

const routes = [
    { path: "/dashboard", name: 'Dashboard', component: Dashboard },
    { path: "/form", name: 'FormInput', component: FormInput },
]

export const router = createRouter({
    history: createMemoryHistory(),
    routes,
})

export default router;