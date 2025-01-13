import React from 'react';
import { FileInput, rem } from '@mantine/core';
import { IconFileCv } from '@tabler/icons-react';

import apiClient from "../service/apiClient.ts";

const Home: React.FC = () => {
    const icon = <IconFileCv style={{ width: rem(18), height: rem(18) }} stroke={1.5} />;
     const handleFileUpload = async (file: File | null) => {
            if (!file) return; // Ensure a file is selected

            const formData = new FormData();
            formData.append('file', file);

            try {
                const response = await apiClient.post('/process_image', formData);
                console.log('File uploaded successfully:', response.data);
            } catch (error) {
                console.error('Error uploading file:', error);
            }
        };

    return (
        <>
            <FileInput
                accept="image/png,image/jpeg"
                leftSection={icon}
                label="Attach your CV"
                placeholder="Your CV"
                leftSectionPointerEvents="none"
                onChange={handleFileUpload}
            />
        </>
    );
}

export default Home;
