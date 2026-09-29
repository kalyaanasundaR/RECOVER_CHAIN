import { useEffect, useRef, useState } from 'react';
import { UploadCloud } from 'lucide-react';
import { StepProps } from '../types';
import { uploadData, analyzeData, getImport, getImportStatus, previewData } from '../../lib/api';
import { ErrorNote, Note } from '../../ui';

export default function S01Upload({ patch, next, setAction }: StepProps) {
    const [file, setFile] = useState<File | null>(null);
    const [err, setErr] = useState('');
    const input = useRef<HTMLInputElement>(null);
    const [drag, setDrag] = useState(false);

    async function analyze(f: File) {
        setErr('');
        try {
            const { dataset_id } = await uploadData(f);
            await analyzeData(dataset_id);
            const [detail, status, preview] = await Promise.all([
                getImport(dataset_id),
                getImportStatus(dataset_id),
                previewData(dataset_id, 6),
            ]);
            patch({ importId: dataset_id, detail, status, preview });
            next();
        } catch (e: any) {
            setErr(e.message);
        }
    }

    async function sample() {
        setErr('');
        try {
            const blob = await fetch('/sample.csv').then((r) => r.blob());
            await analyze(new File([blob], 'sample_payments.csv', { type: 'text/csv' }));
        } catch (e: any) {
            setErr(e.message);
        }
    }

    useEffect(() => {
        setAction(
            file
                ? {
                    label: 'Analyze data →',
                    busy: 'Reading and profiling your file…',
                    onClick: () => analyze(file),
                }
                : null,
        );
    }, [file]);

    return (
        <div className="max-w-xl">
            <h1 className="text-3xl font-bold tracking-tight">Start with your revenue data</h1>
            <p className="mt-3 text-[--muted]">
                Upload a file of failed or overdue payments — one row per payment. RecoverChain
                reads it, works out which revenue is at risk, and decides the safest way to recover
                each one.
            </p>

            <label
                onDragOver={(e) => {
                    e.preventDefault();
                    setDrag(true);
                }}
                onDragLeave={() => setDrag(false)}
                onDrop={(e) => {
                    e.preventDefault();
                    setDrag(false);
                    const f = e.dataTransfer.files?.[0];
                    if (f) setFile(f);
                }}
                className={`mt-8 flex cursor-pointer flex-col items-center gap-3 rounded-2xl border-2 border-dashed p-12 text-center transition-colors ${drag
                    ? 'border-[--accent] bg-sky-500/[0.06]'
                    : 'border-[--line] hover:border-slate-600'
                    }`}
            >
                <UploadCloud className="text-[--muted]" size={28} />
                <span className="font-medium">
                    {file ? file.name : 'Drop a file here, or click to choose'}
                </span>
                <span className="text-xs text-[--faint]">XLSX</span>
                <input
                    ref={input}
                    type="file"
                    accept=".csv,.parquet,.xlsx,.zip"
                    className="hidden"
                    onChange={(e) => setFile(e.target.files?.[0] ?? null)}
                />
            </label>

            <div className="mt-4 text-sm text-[--faint]">
                No file handy?{' '}
                <button onClick={sample} className="font-medium text-sky-400 hover:underline">
                    Use sample data
                </button>
            </div>

            {err && (
                <div className="mt-5">
                    <ErrorNote onRetry={() => file && analyze(file)}>{err}</ErrorNote>
                </div>
            )}

            <div className="mt-8">
                <Note>
                    Formats accepted by the current backend: XLSX. Max
                    500&nbsp;MB.
                </Note>
            </div>
        </div>
    );
}
