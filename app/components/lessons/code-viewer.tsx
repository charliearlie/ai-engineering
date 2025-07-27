'use client';

import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { oneDark, oneLight } from 'react-syntax-highlighter/dist/cjs/styles/prism';
import { useTheme } from 'next-themes';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader } from '@/components/ui/card';
import { Copy, Check, Download, Play, FileText } from 'lucide-react';
import { useState } from 'react';
import { toast } from 'sonner';
import { cn } from '@/lib/utils';

interface CodeViewerProps {
  code: string;
  language?: string;
  filename?: string;
  showLineNumbers?: boolean;
  className?: string;
}

export function CodeViewer({ 
  code, 
  language = 'python',
  filename = 'code.py',
  showLineNumbers = true,
  className 
}: CodeViewerProps) {
  const { theme } = useTheme();
  const [copied, setCopied] = useState(false);

  const copyToClipboard = async () => {
    try {
      await navigator.clipboard.writeText(code);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
      toast.success('Code copied to clipboard!');
    } catch (error) {
      toast.error('Failed to copy code');
    }
  };

  const downloadCode = () => {
    const blob = new Blob([code], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    toast.success('Code downloaded!');
  };

  const runCode = () => {
    toast.info('Code execution feature coming soon!', {
      description: 'We\'re working on a sandboxed Python environment for you to run code safely.',
    });
  };

  const getLanguageIcon = () => {
    switch (language.toLowerCase()) {
      case 'python':
        return '🐍';
      case 'javascript':
      case 'js':
        return '⚡';
      case 'typescript':
      case 'ts':
        return '📘';
      default:
        return '📄';
    }
  };

  return (
    <Card className={cn('w-full', className)}>
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <div className="flex items-center gap-2 text-sm font-medium">
              <span className="text-lg">{getLanguageIcon()}</span>
              <FileText className="w-4 h-4" />
              <span>{filename}</span>
            </div>
            <div className="text-xs text-muted-foreground px-2 py-1 bg-muted rounded">
              {language.toUpperCase()}
            </div>
          </div>

          <div className="flex items-center gap-2">
            <Button
              size="sm"
              variant="outline"
              onClick={runCode}
              className="h-8"
            >
              <Play className="w-4 h-4 mr-1" />
              Run Code
            </Button>
            
            <Button
              size="sm"
              variant="outline"
              onClick={downloadCode}
              className="h-8"
            >
              <Download className="w-4 h-4 mr-1" />
              Download
            </Button>
            
            <Button
              size="sm"
              variant="outline"
              onClick={copyToClipboard}
              className="h-8"
            >
              {copied ? (
                <Check className="w-4 h-4 mr-1 text-green-500" />
              ) : (
                <Copy className="w-4 h-4 mr-1" />
              )}
              {copied ? 'Copied!' : 'Copy'}
            </Button>
          </div>
        </div>
      </CardHeader>

      <CardContent className="p-0">
        <div className="relative">
          <SyntaxHighlighter
            style={theme === 'dark' ? oneDark : oneLight}
            language={language}
            showLineNumbers={showLineNumbers}
            customStyle={{
              margin: 0,
              borderRadius: '0 0 8px 8px',
              fontSize: '14px',
              lineHeight: '1.5',
            }}
            codeTagProps={{
              style: {
                fontFamily: 'var(--font-geist-mono), ui-monospace, SFMono-Regular, "SF Mono", Consolas, "Liberation Mono", Menlo, monospace',
              }
            }}
          >
            {code}
          </SyntaxHighlighter>
        </div>
      </CardContent>
    </Card>
  );
}