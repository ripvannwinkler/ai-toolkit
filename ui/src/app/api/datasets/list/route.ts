import { NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';
import { getDatasetsRoot } from '@/server/settings';
import { findImagesRecursively } from '@/server/images';

export async function GET() {
  try {
    let datasetsPath = await getDatasetsRoot();

    // if folder doesnt exist, create it
    try {
      await fs.promises.access(datasetsPath);
    } catch {
      await fs.promises.mkdir(datasetsPath);
    }

    // find all the folders in the datasets folder
    let folders = (await fs.promises.readdir(datasetsPath, { withFileTypes: true }))
      .filter(dirent => dirent.isDirectory())
      .filter(dirent => !dirent.name.startsWith('.'))
      .map(dirent => dirent.name);

    // Count the total number of images in each dataset so the list screen can
    // show per-dataset totals without a second round-trip. Counts are gathered
    // concurrently to avoid serializing the (potentially slow) recursive walk.
    const datasets = await Promise.all(
      folders.map(async name => {
        const folder = path.join(datasetsPath, name);
        let count = 0;
        try {
          count = (await findImagesRecursively(folder)).length;
        } catch (err) {
          // An unreadable dataset shouldn't break the whole list — report 0.
          console.error(`Failed to count images for dataset '${name}':`, err);
        }
        return { name, count };
      }),
    );

    return NextResponse.json(datasets);
  } catch (error) {
    return NextResponse.json({ error: 'Failed to fetch datasets' }, { status: 500 });
  }
}
