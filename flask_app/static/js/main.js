/**
 * main.js
 * =======
 * Interactive preset loader and form logic for CrediPulse AI Flask application.
 */

const PRESETS = {
  low: {
    Age: 48,
    Education: "Master's",
    MaritalStatus: "Married",
    HasDependents: "Yes",
    EmploymentType: "Full-time",
    MonthsEmployed: 96,
    Income: 98000,
    DTIRatio: 0.18,
    CreditScore: 780,
    NumCreditLines: 4,
    HasMortgage: "Yes",
    HasCoSigner: "Yes",
    LoanAmount: 30000,
    InterestRate: 4.8,
    LoanTerm: "36",
    LoanPurpose: "Home"
  },
  med: {
    Age: 32,
    Education: "Bachelor's",
    MaritalStatus: "Single",
    HasDependents: "No",
    EmploymentType: "Full-time",
    MonthsEmployed: 24,
    Income: 45000,
    DTIRatio: 0.42,
    CreditScore: 620,
    NumCreditLines: 5,
    HasMortgage: "No",
    HasCoSigner: "No",
    LoanAmount: 35000,
    InterestRate: 14.5,
    LoanTerm: "48",
    LoanPurpose: "Auto"
  },
  high: {
    Age: 21,
    Education: "High School",
    MaritalStatus: "Single",
    HasDependents: "No",
    EmploymentType: "Unemployed",
    MonthsEmployed: 3,
    Income: 16000,
    DTIRatio: 0.78,
    CreditScore: 410,
    NumCreditLines: 8,
    HasMortgage: "No",
    HasCoSigner: "No",
    LoanAmount: 95000,
    InterestRate: 22.8,
    LoanTerm: "60",
    LoanPurpose: "Business"
  }
};

function fillPreset(type) {
  const profile = PRESETS[type];
  if (!profile) return;

  for (const [key, value] of Object.entries(profile)) {
    const el = document.getElementById(key);
    if (el) {
      el.value = value;
      // Trigger subtle highlight animation
      el.style.borderColor = "var(--accent-cyan)";
      setTimeout(() => {
        el.style.borderColor = "";
      }, 500);
    }
  }
}
