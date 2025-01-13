import React from 'react';
import { FileInput, rem } from '@mantine/core';
import { IconFileCv } from '@tabler/icons-react';

const Home: React.FC = () => {
    const icon = <IconFileCv style={{ width: rem(18), height: rem(18) }} stroke={1.5} />;

    return (
        <>
            <FileInput
                leftSection={icon}
                label="Attach your CV"
                placeholder="Your CV"
                leftSectionPointerEvents="none"
            />
            <FileInput
                rightSection={icon}
                label="Attach your CV"
                placeholder="Your CV"
                rightSectionPointerEvents="none"
                mt="md"
            />
        </>
    );
}

export default Home;
