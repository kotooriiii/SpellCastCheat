import axios from 'axios';

const baseClient = axios.create({
    baseURL: 'http://localhost:8080',
    timeout: 20000,
    headers: {
        'Content-Type': 'multipart/form-data',
        'X-Custom-Header': "use-preflight",
    },
});
export default baseClient;
