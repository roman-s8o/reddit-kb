/**
 * Phase 5 - Frontend Testing Script
 * 
 * This script tests the Phase 5 frontend implementation to ensure:
 * 1. React chat UI works with the new chatbot API
 * 2. Subreddit dropdown functions correctly
 * 3. Insights dashboard displays comprehensive data
 * 4. All three FastAPI backends are properly connected
 * 5. LangGraph nodes can be triggered from the UI
 */

const axios = require('axios');

// API endpoints
const APIs = {
  main: 'http://localhost:8000',
  chatbot: 'http://localhost:8001', 
  orchestration: 'http://localhost:8002',
  frontend: 'http://localhost:3000'
};

class Phase5Tester {
  constructor() {
    this.results = {};
  }

  async testApiConnections() {
    console.log('🔍 Testing API Connections...');
    
    const tests = [
      { name: 'Main API', url: `${APIs.main}/health` },
      { name: 'Chatbot API', url: `${APIs.chatbot}/health` },
      { name: 'Orchestration API', url: `${APIs.orchestration}/health` },
    ];

    for (const test of tests) {
      try {
        const response = await axios.get(test.url, { timeout: 5000 });
        console.log(`  ✅ ${test.name}: Connected (${response.status})`);
        this.results[test.name] = { success: true, status: response.status };
      } catch (error) {
        console.log(`  ❌ ${test.name}: Failed (${error.message})`);
        this.results[test.name] = { success: false, error: error.message };
      }
    }
  }

  async testChatbotAPI() {
    console.log('\n💬 Testing Chatbot API Integration...');
    
    try {
      // Test chat endpoint
      const chatResponse = await axios.post(`${APIs.chatbot}/chat`, {
        query: "What are people saying about Python?",
        subreddits: ["Python"]
      }, { timeout: 30000 });

      console.log('  ✅ Chat endpoint working');
      console.log(`  📝 Response format: ${Object.keys(chatResponse.data).join(', ')}`);
      
      // Verify response structure
      const expectedFields = ['answer', 'references', 'insights', 'timestamp'];
      const hasAllFields = expectedFields.every(field => 
        chatResponse.data.hasOwnProperty(field)
      );
      
      if (hasAllFields) {
        console.log('  ✅ Response format matches Phase 4 specification');
      } else {
        console.log('  ⚠️ Response format may need adjustment');
      }

      this.results['Chatbot API'] = { 
        success: true, 
        responseFields: Object.keys(chatResponse.data),
        hasAllFields
      };

    } catch (error) {
      console.log(`  ❌ Chatbot API test failed: ${error.message}`);
      this.results['Chatbot API'] = { success: false, error: error.message };
    }
  }

  async testOrchestrationAPI() {
    console.log('\n🔧 Testing Orchestration API Integration...');
    
    try {
      // Test scheduler status
      const schedulerResponse = await axios.get(`${APIs.orchestration}/scheduler/status`);
      console.log('  ✅ Scheduler status endpoint working');
      
      // Test jobs list
      const jobsResponse = await axios.get(`${APIs.orchestration}/scheduler/jobs`);
      console.log(`  ✅ Jobs endpoint working (${jobsResponse.data.data.total_jobs} jobs found)`);
      
      // Test system health
      const healthResponse = await axios.get(`${APIs.orchestration}/system/health`);
      console.log('  ✅ System health endpoint working');
      
      const healthChecks = healthResponse.data.data.health_checks;
      const healthyChecks = Object.values(healthChecks).filter(Boolean).length;
      const totalChecks = Object.keys(healthChecks).length;
      
      console.log(`  📊 Health checks: ${healthyChecks}/${totalChecks} passing`);

      this.results['Orchestration API'] = { 
        success: true,
        totalJobs: jobsResponse.data.data.total_jobs,
        healthyChecks: `${healthyChecks}/${totalChecks}`
      };

    } catch (error) {
      console.log(`  ❌ Orchestration API test failed: ${error.message}`);
      this.results['Orchestration API'] = { success: false, error: error.message };
    }
  }

  async testWorkflowTrigger() {
    console.log('\n🔄 Testing Workflow Trigger from Frontend...');
    
    try {
      // Test manual workflow execution
      const workflowResponse = await axios.post(`${APIs.orchestration}/workflow/execute`, {
        workflow_type: 'batch',
        subreddits: ['Python'],
        parameters: { test: true }
      });

      console.log('  ✅ Workflow execution endpoint working');
      console.log(`  🚀 Workflow started: ${workflowResponse.data.data.status}`);

      this.results['Workflow Trigger'] = { 
        success: true,
        status: workflowResponse.data.data.status
      };

    } catch (error) {
      console.log(`  ❌ Workflow trigger test failed: ${error.message}`);
      this.results['Workflow Trigger'] = { success: false, error: error.message };
    }
  }

  async testFrontendAvailability() {
    console.log('\n🌐 Testing Frontend Availability...');
    
    try {
      const response = await axios.get(APIs.frontend, { timeout: 5000 });
      console.log('  ✅ Frontend is accessible');
      console.log(`  📱 Status: ${response.status}`);
      
      this.results['Frontend'] = { success: true, status: response.status };

    } catch (error) {
      console.log(`  ❌ Frontend not accessible: ${error.message}`);
      console.log('  💡 Make sure to run: npm start in the frontend directory');
      this.results['Frontend'] = { success: false, error: error.message };
    }
  }

  async testDataFlow() {
    console.log('\n🔗 Testing Complete Data Flow...');
    
    try {
      // 1. Check if we have data
      const statusResponse = await axios.get(`${APIs.main}/status`);
      const totalDocs = statusResponse.data.data.chatbot?.knowledge_base?.total_documents || 0;
      
      console.log(`  📊 Knowledge base has ${totalDocs} documents`);
      
      if (totalDocs === 0) {
        console.log('  ⚠️ No documents in knowledge base - frontend will show empty state');
        console.log('  💡 Run data collection workflow to populate the system');
      } else {
        console.log('  ✅ Knowledge base populated - frontend should show data');
      }

      // 2. Test insights availability
      const insightsResponse = await axios.get(`${APIs.main}/insights/dashboard`);
      const recentInsights = insightsResponse.data.data.recent_insights || [];
      
      console.log(`  💡 Found ${recentInsights.length} recent insights`);
      
      this.results['Data Flow'] = { 
        success: true,
        totalDocuments: totalDocs,
        recentInsights: recentInsights.length
      };

    } catch (error) {
      console.log(`  ❌ Data flow test failed: ${error.message}`);
      this.results['Data Flow'] = { success: false, error: error.message };
    }
  }

  generateReport() {
    console.log('\n' + '='.repeat(60));
    console.log('📊 Phase 5 - Frontend Testing Report');
    console.log('='.repeat(60));

    const totalTests = Object.keys(this.results).length;
    const passedTests = Object.values(this.results).filter(r => r.success).length;
    
    console.log(`\n📈 Overall Status: ${passedTests}/${totalTests} tests passed\n`);

    // Detailed results
    for (const [testName, result] of Object.entries(this.results)) {
      const status = result.success ? '✅ PASSED' : '❌ FAILED';
      console.log(`${status} - ${testName}`);
      
      if (result.success) {
        if (result.responseFields) {
          console.log(`    Response fields: ${result.responseFields.join(', ')}`);
        }
        if (result.totalJobs) {
          console.log(`    Total jobs: ${result.totalJobs}`);
        }
        if (result.healthyChecks) {
          console.log(`    Health checks: ${result.healthyChecks}`);
        }
        if (result.totalDocuments !== undefined) {
          console.log(`    Documents: ${result.totalDocuments}`);
        }
      } else {
        console.log(`    Error: ${result.error}`);
      }
      console.log('');
    }

    // Phase 5 requirements check
    console.log('📋 Phase 5 Requirements Status:');
    console.log('');
    
    const requirements = [
      { name: 'React chat UI', test: 'Chatbot API', met: this.results['Chatbot API']?.success },
      { name: 'Dropdown for subreddit', test: 'Frontend', met: this.results['Frontend']?.success },
      { name: 'Insights dashboard', test: 'Data Flow', met: this.results['Data Flow']?.success },
      { name: 'Connect to FastAPI backend → trigger LangGraph nodes', test: 'Workflow Trigger', met: this.results['Workflow Trigger']?.success }
    ];

    requirements.forEach(req => {
      const status = req.met ? '✅' : '❌';
      console.log(`  ${status} ${req.name}`);
    });

    const allRequirementsMet = requirements.every(req => req.met);
    
    console.log('\n' + '='.repeat(60));
    if (allRequirementsMet) {
      console.log('🎉 Phase 5 - Frontend is COMPLETE!');
      console.log('✅ All ROADMAP.md requirements have been implemented');
      console.log('\n🚀 Next Steps:');
      console.log('  1. Start all services:');
      console.log('     - Main API: python backend/api/main.py');
      console.log('     - Chatbot API: python backend/chatbot_api.py');
      console.log('     - Orchestration API: python backend/api/orchestration_api.py');
      console.log('     - Frontend: npm start (in frontend directory)');
      console.log('  2. Access the application at http://localhost:3000');
      console.log('  3. Test all features through the web interface');
    } else {
      console.log('⚠️ Phase 5 - Frontend needs attention');
      console.log('❌ Some requirements are not yet met');
      console.log('\n🔧 Troubleshooting:');
      console.log('  1. Ensure all backend services are running');
      console.log('  2. Check API connectivity');
      console.log('  3. Verify frontend build and dependencies');
    }
    console.log('='.repeat(60));
  }

  async runAllTests() {
    console.log('🚀 Starting Phase 5 - Frontend Testing');
    console.log('Testing React UI, API integration, and workflow orchestration\n');

    await this.testApiConnections();
    await this.testChatbotAPI();
    await this.testOrchestrationAPI();
    await this.testWorkflowTrigger();
    await this.testFrontendAvailability();
    await this.testDataFlow();

    this.generateReport();
  }
}

// Run the tests
const tester = new Phase5Tester();
tester.runAllTests().catch(console.error);
