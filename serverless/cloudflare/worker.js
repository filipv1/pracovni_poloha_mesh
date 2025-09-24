/**
 * Cloudflare Worker - Orchestrator for Ergonomic Analysis Pipeline
 * Handles auth, RunPod job submission, storage, emails, and logging
 */

// Configuration (will be set as environment variables)
const CONFIG = {
  RUNPOD_API_KEY: '', // Set in Cloudflare dashboard
  RUNPOD_ENDPOINT_ID: '', // Set after deploying to RunPod
  RESEND_API_KEY: '', // From Resend.com
  AUTH_PASSWORD_HASH: '', // bcrypt hash of password
  FRONTEND_URL: 'https://yourusername.github.io/ergonomic-analyzer'
};

// CORS headers for frontend
const CORS_HEADERS = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
  'Access-Control-Allow-Headers': 'Content-Type, Authorization',
  'Access-Control-Max-Age': '86400',
};

export default {
  async fetch(request, env, ctx) {
    // Handle CORS preflight
    if (request.method === 'OPTIONS') {
      return new Response(null, { headers: CORS_HEADERS });
    }

    const url = new URL(request.url);
    const path = url.pathname;

    try {
      // Route requests
      switch (path) {
        case '/auth/login':
          return handleLogin(request, env);

        case '/jobs/submit':
          return handleJobSubmit(request, env, ctx);

        case '/jobs/status':
          return handleJobStatus(request, env);

        case '/jobs/result':
          return handleJobResult(request, env);

        case '/health':
          return handleHealth(env);

        default:
          return new Response('Not Found', { status: 404 });
      }
    } catch (error) {
      console.error('Worker error:', error);
      return new Response(JSON.stringify({ error: error.message }), {
        status: 500,
        headers: { ...CORS_HEADERS, 'Content-Type': 'application/json' }
      });
    }
  }
};

// Authentication handler
async function handleLogin(request, env) {
  const body = await request.json();
  const { password } = body;

  // Simple password check (in production, use proper bcrypt comparison)
  const validPassword = await verifyPassword(password, env.AUTH_PASSWORD_HASH);

  if (validPassword) {
    // Generate simple JWT token
    const token = await generateJWT({ authenticated: true }, env.JWT_SECRET);

    return new Response(JSON.stringify({ token }), {
      headers: { ...CORS_HEADERS, 'Content-Type': 'application/json' }
    });
  }

  return new Response('Unauthorized', {
    status: 401,
    headers: CORS_HEADERS
  });
}

// Job submission handler
async function handleJobSubmit(request, env, ctx) {
  // Verify auth
  const token = request.headers.get('Authorization')?.replace('Bearer ', '');
  if (!token || !await verifyJWT(token, env.JWT_SECRET)) {
    return new Response('Unauthorized', { status: 401, headers: CORS_HEADERS });
  }

  const formData = await request.formData();
  const videoFile = formData.get('video');
  const userEmail = formData.get('email');
  const quality = formData.get('quality') || 'medium';

  if (!videoFile) {
    return new Response('No video file provided', { status: 400, headers: CORS_HEADERS });
  }

  // Convert video to base64
  const videoBuffer = await videoFile.arrayBuffer();
  const videoBase64 = btoa(String.fromCharCode(...new Uint8Array(videoBuffer)));

  // Generate job ID
  const jobId = crypto.randomUUID();

  // Store job metadata in KV
  await env.JOBS.put(`job:${jobId}`, JSON.stringify({
    id: jobId,
    status: 'queued',
    email: userEmail,
    filename: videoFile.name,
    timestamp: Date.now(),
    quality: quality
  }), { expirationTtl: 172800 }); // 48 hours

  // Submit to RunPod
  ctx.waitUntil(submitToRunPod(jobId, videoBase64, videoFile.name, quality, userEmail, env));

  // Log job submission
  await logUsage(env, userEmail, 'job_submitted', { jobId, filename: videoFile.name });

  return new Response(JSON.stringify({
    jobId,
    message: 'Job submitted successfully'
  }), {
    headers: { ...CORS_HEADERS, 'Content-Type': 'application/json' }
  });
}

// Submit job to RunPod
async function submitToRunPod(jobId, videoBase64, filename, quality, userEmail, env) {
  try {
    // Update status to processing
    await updateJobStatus(env, jobId, 'processing', { runpod_status: 'submitting' });

    const response = await fetch(`https://api.runpod.ai/v2/${env.RUNPOD_ENDPOINT_ID}/run`, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${env.RUNPOD_API_KEY}`,
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        input: {
          video_base64: videoBase64,
          video_name: filename,
          quality: quality,
          user_email: userEmail
        },
        webhook: `${env.WORKER_URL}/webhook/runpod`
      })
    });

    const result = await response.json();

    if (result.id) {
      // Store RunPod job ID mapping
      await env.JOBS.put(`runpod:${result.id}`, jobId, { expirationTtl: 172800 });
      await updateJobStatus(env, jobId, 'processing', {
        runpod_id: result.id,
        runpod_status: 'running'
      });
    } else {
      throw new Error('Failed to submit to RunPod');
    }

    // Start polling for results
    pollRunPodStatus(env, jobId, result.id);

  } catch (error) {
    console.error('RunPod submission error:', error);
    await updateJobStatus(env, jobId, 'failed', { error: error.message });

    // Send failure email
    await sendEmail(env, userEmail, 'failed', { jobId, error: error.message });
  }
}

// Poll RunPod job status
async function pollRunPodStatus(env, jobId, runpodId) {
  const maxAttempts = 60; // 10 minutes max
  let attempts = 0;

  const poll = async () => {
    try {
      const response = await fetch(`https://api.runpod.ai/v2/${env.RUNPOD_ENDPOINT_ID}/status/${runpodId}`, {
        headers: {
          'Authorization': `Bearer ${env.RUNPOD_API_KEY}`
        }
      });

      const status = await response.json();

      if (status.status === 'COMPLETED') {
        // Get results
        const output = status.output;

        // Store results in R2
        const pklKey = `results/${jobId}/analysis.pkl`;
        const xlsxKey = `results/${jobId}/analysis.xlsx`;

        await env.R2.put(pklKey, base64ToArrayBuffer(output.pkl_base64), {
          customMetadata: { jobId, type: 'pkl' }
        });

        await env.R2.put(xlsxKey, base64ToArrayBuffer(output.xlsx_base64), {
          customMetadata: { jobId, type: 'xlsx' }
        });

        // Update job status
        await updateJobStatus(env, jobId, 'completed', {
          pkl_url: pklKey,
          xlsx_url: xlsxKey,
          statistics: output.statistics
        });

        // Send success email
        const jobData = JSON.parse(await env.JOBS.get(`job:${jobId}`));
        await sendEmail(env, jobData.email, 'completed', {
          jobId,
          downloadUrl: `${env.WORKER_URL}/jobs/result?id=${jobId}`
        });

        // Log completion
        await logUsage(env, jobData.email, 'job_completed', {
          jobId,
          duration: Date.now() - jobData.timestamp
        });

      } else if (status.status === 'FAILED') {
        await updateJobStatus(env, jobId, 'failed', { error: status.error });

        const jobData = JSON.parse(await env.JOBS.get(`job:${jobId}`));
        await sendEmail(env, jobData.email, 'failed', { jobId, error: status.error });

      } else if (attempts < maxAttempts) {
        // Continue polling
        attempts++;
        setTimeout(poll, 10000); // Poll every 10 seconds
      } else {
        // Timeout
        await updateJobStatus(env, jobId, 'failed', { error: 'Processing timeout' });
      }

    } catch (error) {
      console.error('Polling error:', error);
      await updateJobStatus(env, jobId, 'failed', { error: error.message });
    }
  };

  // Start polling
  poll();
}

// Get job status
async function handleJobStatus(request, env) {
  const url = new URL(request.url);
  const jobId = url.searchParams.get('id');

  if (!jobId) {
    return new Response('Job ID required', { status: 400, headers: CORS_HEADERS });
  }

  const jobData = await env.JOBS.get(`job:${jobId}`);

  if (!jobData) {
    return new Response('Job not found', { status: 404, headers: CORS_HEADERS });
  }

  return new Response(jobData, {
    headers: { ...CORS_HEADERS, 'Content-Type': 'application/json' }
  });
}

// Get job results
async function handleJobResult(request, env) {
  const url = new URL(request.url);
  const jobId = url.searchParams.get('id');
  const type = url.searchParams.get('type') || 'xlsx'; // xlsx or pkl

  if (!jobId) {
    return new Response('Job ID required', { status: 400, headers: CORS_HEADERS });
  }

  const jobData = JSON.parse(await env.JOBS.get(`job:${jobId}`));

  if (!jobData || jobData.status !== 'completed') {
    return new Response('Results not available', { status: 404, headers: CORS_HEADERS });
  }

  // Get file from R2
  const fileKey = type === 'pkl' ? jobData.pkl_url : jobData.xlsx_url;
  const object = await env.R2.get(fileKey);

  if (!object) {
    return new Response('File not found', { status: 404, headers: CORS_HEADERS });
  }

  // Return file with appropriate headers
  const extension = type === 'pkl' ? 'pkl' : 'xlsx';
  const mimeType = type === 'pkl' ? 'application/octet-stream' :
    'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet';

  return new Response(object.body, {
    headers: {
      ...CORS_HEADERS,
      'Content-Type': mimeType,
      'Content-Disposition': `attachment; filename="analysis_${jobId}.${extension}"`
    }
  });
}

// Health check
async function handleHealth(env) {
  const checks = {
    worker: 'healthy',
    kv: false,
    r2: false,
    runpod: false
  };

  // Check KV
  try {
    await env.JOBS.get('health:check');
    checks.kv = true;
  } catch (e) {}

  // Check R2
  try {
    await env.R2.head('health/check');
    checks.r2 = true;
  } catch (e) {}

  // Check RunPod
  try {
    const response = await fetch(`https://api.runpod.ai/v2/${env.RUNPOD_ENDPOINT_ID}/health`, {
      headers: { 'Authorization': `Bearer ${env.RUNPOD_API_KEY}` }
    });
    checks.runpod = response.ok;
  } catch (e) {}

  const allHealthy = Object.values(checks).every(v => v === true || v === 'healthy');

  return new Response(JSON.stringify(checks), {
    status: allHealthy ? 200 : 503,
    headers: { ...CORS_HEADERS, 'Content-Type': 'application/json' }
  });
}

// Helper: Update job status
async function updateJobStatus(env, jobId, status, additionalData = {}) {
  const existing = JSON.parse(await env.JOBS.get(`job:${jobId}`) || '{}');
  const updated = {
    ...existing,
    ...additionalData,
    status,
    lastUpdated: Date.now()
  };
  await env.JOBS.put(`job:${jobId}`, JSON.stringify(updated), { expirationTtl: 172800 });
}

// Helper: Send email via Resend
async function sendEmail(env, to, type, data) {
  const subjects = {
    completed: 'Your Ergonomic Analysis is Ready',
    failed: 'Ergonomic Analysis Failed'
  };

  const htmlTemplates = {
    completed: `
      <h2>Analysis Complete!</h2>
      <p>Your ergonomic analysis has been completed successfully.</p>
      <p><a href="${data.downloadUrl}" style="padding: 10px 20px; background: #4CAF50; color: white; text-decoration: none; border-radius: 5px;">Download Results</a></p>
      <p>Results will be available for 48 hours.</p>
    `,
    failed: `
      <h2>Analysis Failed</h2>
      <p>Unfortunately, your analysis could not be completed.</p>
      <p>Error: ${data.error}</p>
      <p>Please try again or contact support if the issue persists.</p>
    `
  };

  try {
    await fetch('https://api.resend.com/emails', {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${env.RESEND_API_KEY}`,
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        from: 'Ergonomic Analyzer <noreply@yourdomain.com>',
        to: [to],
        subject: subjects[type],
        html: htmlTemplates[type]
      })
    });
  } catch (error) {
    console.error('Email send error:', error);
  }
}

// Helper: Log usage
async function logUsage(env, user, action, metadata) {
  const logEntry = {
    user,
    action,
    metadata,
    timestamp: Date.now()
  };

  const logKey = `log:${Date.now()}:${crypto.randomUUID()}`;
  await env.LOGS.put(logKey, JSON.stringify(logEntry), { expirationTtl: 2592000 }); // 30 days
}

// Helper: Simple JWT functions
async function generateJWT(payload, secret) {
  const header = btoa(JSON.stringify({ alg: 'HS256', typ: 'JWT' }));
  const body = btoa(JSON.stringify({ ...payload, exp: Date.now() + 86400000 })); // 24h
  const signature = await crypto.subtle.digest('SHA-256',
    new TextEncoder().encode(`${header}.${body}.${secret}`)
  );
  return `${header}.${body}.${btoa(String.fromCharCode(...new Uint8Array(signature)))}`;
}

async function verifyJWT(token, secret) {
  try {
    const [header, body, signature] = token.split('.');
    const payload = JSON.parse(atob(body));

    if (payload.exp < Date.now()) return false;

    const expectedSignature = await crypto.subtle.digest('SHA-256',
      new TextEncoder().encode(`${header}.${body}.${secret}`)
    );

    return signature === btoa(String.fromCharCode(...new Uint8Array(expectedSignature)));
  } catch {
    return false;
  }
}

// Helper: Password verification
async function verifyPassword(password, hash) {
  // In production, use proper bcrypt library
  // For now, simple comparison
  return password === hash;
}

// Helper: Base64 to ArrayBuffer
function base64ToArrayBuffer(base64) {
  const binaryString = atob(base64);
  const bytes = new Uint8Array(binaryString.length);
  for (let i = 0; i < binaryString.length; i++) {
    bytes[i] = binaryString.charCodeAt(i);
  }
  return bytes.buffer;
}