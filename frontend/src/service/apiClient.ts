// src/services/apiClient.ts
import axios, {CreateAxiosDefaults, InternalAxiosRequestConfig} from 'axios';
import baseClient from './baseClient';

// Clone the configuration of baseClient
const apiClientConfig: CreateAxiosDefaults = {
    ...baseClient.defaults,
};

// Create a new Axios instance with the cloned configuration
const apiClient = axios.create(apiClientConfig);

apiClient.interceptors.request.use((config: InternalAxiosRequestConfig) =>
{
    config.baseURL = config.baseURL + "";
    return config;
});

export default apiClient;


