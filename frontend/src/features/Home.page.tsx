import React, { useState } from 'react';
import { Divider, FileInput, rem } from '@mantine/core';
import { IconFileCv } from '@tabler/icons-react';
import apiClient from "../service/apiClient.ts";


const Home: React.FC = () => {
    const [grid, setGrid] = useState([[]]);
    const [sequences, setSequences] = useState([]);
    const icon = <IconFileCv style={{ width: rem(18), height: rem(18) }} stroke={1.5} />;
    const handleFileUpload = async (file: File | null) => {
        if (!file) return; // Ensure a file is selected

        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await apiClient.post('/process_image', formData);
            console.log('File uploaded successfully:', response.data);
            setGrid(response.data['grid'])
            setSequences(response.data['sequences'])
        } catch (error) {
            console.error('Error uploading file:', error);
        }
    };

    return (
        <>
            <div className='p-2 pb-4'>
                <FileInput
                    accept="image/png,image/jpeg"
                    leftSection={icon}
                    label="Attach your CV"
                    placeholder="Your CV"
                    leftSectionPointerEvents="none"
                    onChange={handleFileUpload}
                />
            </div>

            <Divider />

            <div>
                {
                    sequences.map((e) => {
                        return <div className='p-2'>
                            <div className=''>{'Points: ' + (e.cost)}, {'Word: ' + e.word}</div>
                            <Grid grid={grid} path={e.path}></Grid>
                        </div>
                    })
                }
            </div>
        </>
    );
}

const Grid = ({ grid, path }) => { // path [[x1, y1], [x2, y2], ...]
    const pathSet = new Set(path.map(([x, y]) => `${x},${y}`));    

    return <div className='size-40'>
        <div className='grid grid-cols-5 gap-2'>
            {
                grid.map((rows, x) => rows.map((e, y) => // e.id, e.value, e.x, e.y
                {   
                    const isInPath = pathSet.has(`${x},${y}`);
                    var k = (5 * x) + y
                    return <Box key={k} value={e.letter} isInPath={isInPath}></Box>
                }
                ))
            }
        </div>
    </div>
}

const Box = ({ value, isInPath }) => {
    return isInPath ? <div className='flex justify-center content-center shadow bg-lime-500'>{value}</div> : <div className='flex justify-center content-center shadow'>{value}</div>
}

export default Home;
