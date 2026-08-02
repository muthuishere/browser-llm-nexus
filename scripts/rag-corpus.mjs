// A corpus built to be HARD, because an easy one measures nothing.
//
// Two design rules:
//
// 1. Every question has exactly ONE gold document, and the corpus contains
//    near-miss distractors on the same subject — same vocabulary, same entities,
//    different fact. A corpus of unrelated documents scores recall@5 = 100% on
//    any embedder including a random one, so it tells you nothing.
//
// 2. Every answer is an unguessable token (`RX-####`). A model cannot produce
//    one by knowing the world, so a grounded answer proves retrieval put the
//    right chunk in front of it — not that the model already knew.
//
// The documents are deliberately written in flat, similar prose. Distinctive
// writing style would give an embedder a shortcut that real documents don't.

export const DOCS = [
  // ── Pair 1: identical phrase "refund window", split by standard/express ──
  {
    id: 'refunds-standard',
    title: 'Refund policy — standard orders',
    text: `Refund policy for standard orders. Customers may request a refund on any standard order.
Refund requests are processed by the billing team once the returned item arrives at the warehouse.
The refund window for standard orders is RX-4182 days from delivery.
Refunds are issued to the original payment method and cannot be redirected to a different card.
A refund does not include the original shipping charge unless the item arrived damaged.`,
  },
  {
    id: 'refunds-express',
    title: 'Refund policy — express orders',
    text: `Refund policy for express orders. Customers may request a refund on any express order.
Refund requests are processed by the billing team once the returned item arrives at the warehouse.
The refund window for express orders is RX-9930 days from delivery.
Refunds are issued to the original payment method and cannot be redirected to a different card.
A refund does not include the original shipping charge unless the item arrived damaged.`,
  },

  // ── Pair 2: identical phrase "probation period", split by employee/contractor
  {
    id: 'probation-employees',
    title: 'Probation — permanent employees',
    text: `Probation for permanent employees. New permanent joiners complete orientation in their first week.
Laptops are issued by the IT desk on the first day, after the joining form is signed.
The probation period for permanent employees is RX-2244 days from the start date.
Onboarding buddies are assigned by the reporting manager.
Payroll enrolment happens in the first pay cycle after joining.`,
  },
  {
    id: 'probation-contractors',
    title: 'Probation — contractors',
    text: `Probation for contractors. New contractors complete a shortened orientation on their first day.
Laptops are issued by the IT desk once the contract is countersigned.
The probation period for contractors is RX-6650 days from the start date.
Contractor supervisors are assigned by the engagement owner.
Invoicing begins in the first billing cycle after the engagement starts.`,
  },

  // ── Pair 3: identical phrase "backup retention", split by production/staging
  {
    id: 'backups-production',
    title: 'Backup retention — production',
    text: `Backup policy for production. All production databases are backed up on a fixed schedule.
Backups are encrypted at rest and stored in a separate region from the primary database.
The backup retention period for production databases is RX-8807 days.
Restores are tested by the platform team on a recurring basis.
Backup failures are reported to the platform desk within one working day.`,
  },
  {
    id: 'backups-staging',
    title: 'Backup retention — staging',
    text: `Backup policy for staging. All staging databases are backed up on a fixed schedule.
Backups are encrypted at rest and stored in a separate region from the primary database.
The backup retention period for staging databases is RX-3391 days.
Restores are tested by the platform team on a recurring basis.
Backup failures are reported to the platform desk within one working day.`,
  },

  // ── Pair 4: identical phrase "resolution time", split by severity ──────────
  {
    id: 'incidents-sev1',
    title: 'Incident response — severity one',
    text: `Incident response for severity one. Severity one incidents are triaged by the on-call engineer immediately.
An incident channel is opened for anything affecting more than one customer.
The target resolution time for a severity one incident is RX-6034 minutes.
A written postmortem is published for every severity one incident.
Follow-up actions are tracked to completion by the owning team.`,
  },
  {
    id: 'incidents-sev2',
    title: 'Incident response — severity two',
    text: `Incident response for severity two. Severity two incidents are triaged by the on-call engineer during working hours.
An incident channel is opened for anything affecting more than one customer.
The target resolution time for a severity two incident is RX-1128 minutes.
A postmortem is published at the owning team's discretion for severity two incidents.
Follow-up actions are tracked to completion by the owning team.`,
  },

  // ── Pair 5: identical phrase "expense claim deadline", split by domestic/intl
  {
    id: 'expenses-domestic',
    title: 'Expense claims — domestic',
    text: `Expense claims for domestic travel. Staff may claim reimbursement for approved domestic expenses.
Claims are submitted through the finance portal with a receipt attached.
The expense claim deadline for domestic travel is RX-5573 days after the expense date.
Claims above the manager approval limit are escalated to the finance lead.
Reimbursement is paid with the next payroll run after approval.`,
  },
  {
    id: 'expenses-international',
    title: 'Expense claims — international',
    text: `Expense claims for international travel. Staff may claim reimbursement for approved international expenses.
Claims are submitted through the finance portal with a receipt attached.
The expense claim deadline for international travel is RX-7715 days after the expense date.
Claims above the manager approval limit are escalated to the finance lead.
Reimbursement is paid with the next payroll run after approval.`,
  },
];

// One gold document per question. Every question's key phrase appears in TWO
// documents; only the qualifier (standard/express, production/staging, severity
// one/two) picks the right one. Lexical overlap alone cannot answer these.
export const QUESTIONS = [
  { q: 'How many days is the refund window for standard orders?', gold: 'refunds-standard', answer: 'RX-4182' },
  { q: 'How many days is the refund window for express orders?', gold: 'refunds-express', answer: 'RX-9930' },
  { q: 'How long is the probation period for permanent employees?', gold: 'probation-employees', answer: 'RX-2244' },
  { q: 'How long is the probation period for contractors?', gold: 'probation-contractors', answer: 'RX-6650' },
  { q: 'What is the backup retention period for production databases?', gold: 'backups-production', answer: 'RX-8807' },
  { q: 'What is the backup retention period for staging databases?', gold: 'backups-staging', answer: 'RX-3391' },
  { q: 'What is the target resolution time for a severity one incident?', gold: 'incidents-sev1', answer: 'RX-6034' },
  { q: 'What is the target resolution time for a severity two incident?', gold: 'incidents-sev2', answer: 'RX-1128' },
  { q: 'What is the expense claim deadline for domestic travel?', gold: 'expenses-domestic', answer: 'RX-5573' },
  { q: 'What is the expense claim deadline for international travel?', gold: 'expenses-international', answer: 'RX-7715' },
];
