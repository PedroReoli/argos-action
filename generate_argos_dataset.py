"""
Argos-Action-1.5B: Synthetic Trajectory & MCP Tool Calling Dataset Generator
Author: Pedro Lucas Reis (@pedroreoli)
Repository: pedroreoli/argos-computer-use-mcp
Description: Generates 2,000 rigorous, high-fidelity Computer Use trajectories
mapping Semantic Accessibility Trees (AXTree/UIA) to deterministic Model Context Protocol (MCP) tool calls.
"""

import json
import random
from pathlib import Path

OUTPUT_DIR = Path("F:/Projetos/Reoli/argos-action/data")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TOTAL_SAMPLES = 2000
TRAIN_RATIO = 0.85

SYSTEM_PROMPT = (
    "You are Argos-Action, an expert autonomous Computer Use policy model. "
    "Given the current semantic Accessibility Tree (AXTree/UIA snapshot) of a web or desktop application, "
    "the execution history, and the user's objective, predict the exact next Model Context Protocol (MCP) "
    "tool call to make progress. Output strictly valid JSON with 'thought', 'tool_call', and 'verification_predicate'."
)

# Ambientes & Scenarios Archetypes
ARCHETYPES = [
    # 1. E-Commerce & Retail Navigation
    {
        "domain": "e-commerce",
        "app_names": ["MarketPlace Web", "Storefront Express", "RetailHub"],
        "goals": [
            ("Search for ergonomic mechanical keyboard and add the top-rated item to cart", "keyboard"),
            ("Filter by price under $50, select 4-star rating and apply coupon", "filters"),
            ("Proceed to checkout, select Express Delivery and pay with saved credit card", "checkout"),
            ("Search for organic gluten-free oats, set quantity to 3 and add to cart", "grocery")
        ]
    },
    # 2. Enterprise SaaS, CRM & Analytics
    {
        "domain": "enterprise_saas",
        "app_names": ["CloudCRM Pro", "OpsDashboard", "Enterprise Workspace"],
        "goals": [
            ("Export monthly revenue report as CSV from the billing table", "export_csv"),
            ("Update deal status from 'In Negotiation' to 'Closed Won' for client Acme Corp", "crm_update"),
            ("Create a new priority P0 ticket with title 'API latency degradation' and assign to On-Call", "jira_ticket"),
            ("Filter users table by 'Inactive' status and revoke API keys", "user_management")
        ]
    },
    # 3. Cloud Infrastructure & DevOps
    {
        "domain": "cloud_console",
        "app_names": ["CloudConsole Web", "ClusterControl", "RepoGovernance"],
        "goals": [
            ("Restart the production worker pod 'worker-deployment-78f9c' in cluster us-east-1", "restart_pod"),
            ("Review pull request #142, approve with comment 'LGTM' and merge via squash", "merge_pr"),
            ("Rotate database credentials and download the new secret bundle", "rotate_secrets"),
            ("Scale replica count from 3 to 10 for service 'payment-gateway'", "scale_service")
        ]
    },
    # 4. Native Windows Desktop Applications (UIA)
    {
        "domain": "desktop_windows",
        "app_names": ["Windows File Explorer", "Organon Desktop", "Enterprise Accounting Tool"],
        "goals": [
            ("Navigate to folder 'D:\\Reports\\2026', select file 'audit_q3.xlsx' and click Open", "file_picker"),
            ("Create a new project sprint card named 'Sprint W45' with deadline next Friday", "organon_sprint"),
            ("Switch to Settings window, toggle Dark Theme and save configuration", "settings_toggle"),
            ("Open system print dialog, set 2 copies, double-sided, and send to printer", "print_doc")
        ]
    },
    # 5. Security, 2FA & Edge Cases (Safety & Human Escalation)
    {
        "domain": "security_and_edge_cases",
        "app_names": ["Banking Portal", "SingleSignOn Gateway", "Admin Portal"],
        "goals": [
            ("Handle 2FA verification prompt by requesting human operator input", "mfa_handshake"),
            ("Detect Cloudflare Turnstile/CAPTCHA challenge and request human intervention", "captcha_detected"),
            ("Confirm irreversible account deletion: require explicit human verification", "destructive_action"),
            ("Target element is not found on screen: scroll down to inspect off-screen content", "scroll_discovery")
        ]
    }
]

def generate_ecommerce_trajectory(goal_tuple):
    goal, subcategory = goal_tuple
    
    if subcategory == "keyboard":
        tree = [
            'Window "MarketPlace Web - Shop Electronics" bounds=(0,0,1920,1080)',
            '  [Banner: "Spring Sale 20% off with coupon SPRING20"]',
            '  [Edit: "Search products, brands and categories" id="search_input" value="" focused=false]',
            '  [Button: "Search" id="btn_search" actions=[click]]',
            '  [Button: "Account & Lists" id="btn_account"]',
            '  [Button: "Cart (0)" id="btn_cart"]'
        ]
        history = ["(initial state - navigation landed on home page)"]
        thought = "The user wants to search for an ergonomic mechanical keyboard. The search input is currently empty and unfocused. The first step is typing the query into 'search_input' and submitting."
        tool = "type_text"
        args = {"element_id": "search_input", "text": "ergonomic mechanical keyboard\n", "press_enter": True}
        pred = {"expected_element": "results_container", "timeout_ms": 3500}
        
    elif subcategory == "filters":
        tree = [
            'Window "MarketPlace - Search Results" bounds=(0,0,1920,1080)',
            '  [Heading: "Results for Mechanical Keyboards (2,410 items)"]',
            '  [Group: "Filters Sidebar" id="sidebar_filters"]',
            '    [CheckBox: "Under $50" id="filter_price_under_50" checked=false actions=[click]]',
            '    [CheckBox: "$50 to $100" id="filter_price_50_100" checked=false]',
            '    [CheckBox: "4 Stars & Up" id="filter_rating_4up" checked=false actions=[click]]',
            '    [Edit: "Promo Code" id="input_promo" value=""]',
            '    [Button: "Apply Coupon" id="btn_apply_promo"]'
        ]
        history = ["typed 'mechanical keyboard' into search_input", "results page loaded"]
        thought = "The objective requires filtering by price under $50 and 4-star rating. I must first check the 'Under $50' filter box."
        tool = "click"
        args = {"element_id": "filter_price_under_50", "delivery_mode": "background"}
        pred = {"expected_state": {"filter_price_under_50": "checked=true"}, "timeout_ms": 2000}
        
    elif subcategory == "checkout":
        tree = [
            'Window "Checkout - Shipping & Payment" bounds=(0,0,1920,1080)',
            '  [Heading: "Review Shipping Options"]',
            '  [RadioButton: "Standard Shipping (3-5 business days) - Free" id="radio_std" checked=true]',
            '  [RadioButton: "Express Delivery (Next Day) - $9.99" id="radio_express" checked=false actions=[click]]',
            '  [Heading: "Payment Method"]',
            '  [RadioButton: "Visa ending in 4040" id="radio_card_4040" checked=true]',
            '  [Button: "Place Your Order - $59.98" id="btn_place_order" actions=[click]]'
        ]
        history = ["added item to cart", "proceeded to checkout", "shipping address confirmed"]
        thought = "The user specified 'Express Delivery'. The radio button for Express Delivery is currently unchecked. I will click 'radio_express' before placing the order."
        tool = "click"
        args = {"element_id": "radio_express", "delivery_mode": "background"}
        pred = {"expected_state": {"radio_express": "checked=true"}, "timeout_ms": 1500}
        
    else: # grocery
        tree = [
            'Window "Organic Food Market" bounds=(0,0,1920,1080)',
            '  [Heading: "Organic Rolled Oats Gluten-Free 1kg"]',
            '  [Text: "Certified Gluten-Free, Non-GMO" id="badge_gluten_free"]',
            '  [Combobox: "Quantity" id="select_qty" value="1" options=["1","2","3","4","5"] actions=[select]]',
            '  [Button: "Add to Cart" id="btn_add_to_cart" actions=[click]]',
            '  [Button: "Buy Now" id="btn_buy_now"]'
        ]
        history = ["searched 'organic gluten-free oats'", "selected first product result"]
        thought = "The user wants quantity set to 3. The current quantity selector is at 1. I will select option '3' on 'select_qty'."
        tool = "select_option"
        args = {"element_id": "select_qty", "value": "3"}
        pred = {"expected_state": {"select_qty": "value=3"}, "timeout_ms": 1000}

    return goal, tree, history, thought, tool, args, pred

def generate_saas_trajectory(goal_tuple):
    goal, subcategory = goal_tuple
    
    if subcategory == "export_csv":
        tree = [
            'Window "CloudCRM - Invoices & Revenue" bounds=(0,0,1920,1080)',
            '  [Header: "Billing & Revenue Ledger (November 2026)"]',
            '  [Button: "Filter Date" id="btn_filter_date"]',
            '  [Button: "Export" id="btn_export_dropdown" actions=[click]]',
            '  [Table: "InvoicesTable" rows=150 cols=6]',
            '    [Row 0: "Invoice #INV-2026-001 | Acme Corp | $14,500.00 | Paid"]',
            '    [Row 1: "Invoice #INV-2026-002 | Globex Inc | $8,200.00 | Paid"]'
        ]
        history = ["navigated to Billing module", "selected November 2026 period"]
        thought = "The objective is to export the revenue report as CSV. The Export dropdown button is visible. Clicking 'btn_export_dropdown' will reveal export format options."
        tool = "click"
        args = {"element_id": "btn_export_dropdown", "delivery_mode": "background"}
        pred = {"expected_element": "menu_item_export_csv", "timeout_ms": 1200}
        
    elif subcategory == "crm_update":
        tree = [
            'Window "CloudCRM - Deal: Acme Corp Enterprise Tier" bounds=(0,0,1920,1080)',
            '  [Heading: "Deal Value: $120,000/year"]',
            '  [Group: "Pipeline Stage Progress"]',
            '    [Button: "Lead" active=false]',
            '    [Button: "Discovery" active=false]',
            '    [Button: "In Negotiation" active=true]',
            '    [Button: "Closed Won" id="stage_closed_won" active=false actions=[click]]',
            '    [Button: "Closed Lost" id="stage_closed_lost" active=false]',
            '  [Button: "Save Changes" id="btn_save_deal" disabled=true]'
        ]
        history = ["opened deal card for Acme Corp"]
        thought = "The user wants to mark Acme Corp as 'Closed Won'. I will click the 'stage_closed_won' pill button to transition the pipeline stage."
        tool = "click"
        args = {"element_id": "stage_closed_won", "delivery_mode": "background"}
        pred = {"expected_state": {"stage_closed_won": "active=true", "btn_save_deal": "disabled=false"}, "timeout_ms": 1500}
        
    elif subcategory == "jira_ticket":
        tree = [
            'Window "OpsDashboard - Create Issue" bounds=(0,0,1920,1080)',
            '  [Combobox: "Issue Type" id="input_issue_type" value="Bug"]',
            '  [Edit: "Summary / Title *" id="input_summary" value="" focused=true]',
            '  [Combobox: "Priority" id="input_priority" value="P2 - Medium"]',
            '  [Edit: "Assignee" id="input_assignee" value="Unassigned"]',
            '  [Button: "Create Ticket" id="btn_submit_ticket" disabled=true]'
        ]
        history = ["clicked '+ Create Issue' in navbar"]
        thought = "The modal for ticket creation is open and 'input_summary' is focused. The user requested title 'API latency degradation'. Typing the summary will validate the mandatory field."
        tool = "type_text"
        args = {"element_id": "input_summary", "text": "API latency degradation"}
        pred = {"expected_state": {"input_summary": "value='API latency degradation'"}, "timeout_ms": 1000}
        
    else: # user_management
        tree = [
            'Window "Enterprise Admin - User Access Governance" bounds=(0,0,1920,1080)',
            '  [Combobox: "User Status Filter" id="filter_status" value="All" options=["All", "Active", "Inactive", "Suspended"] actions=[select]]',
            '  [Button: "Bulk Actions" id="btn_bulk_actions" disabled=true]',
            '  [Table: "UserList" total_users=1420]'
        ]
        history = ["navigated to Settings > IAM > User Management"]
        thought = "The user requested filtering users by 'Inactive'. I need to set the status dropdown 'filter_status' to 'Inactive'."
        tool = "select_option"
        args = {"element_id": "filter_status", "value": "Inactive"}
        pred = {"expected_state": {"filter_status": "value='Inactive'"}, "timeout_ms": 2000}

    return goal, tree, history, thought, tool, args, pred

def generate_devops_trajectory(goal_tuple):
    goal, subcategory = goal_tuple
    
    if subcategory == "restart_pod":
        tree = [
            'Window "ClusterControl - Pods: us-east-1 (prod)" bounds=(0,0,1920,1080)',
            '  [SearchBox: "Filter pods by label or name" id="input_pod_filter" value="worker-deployment"]',
            '  [Row: "worker-deployment-78f9c | Status: Running | Restarts: 4 | CPU: 82%"]',
            '    [Button: "View Logs" id="btn_logs_78f9c"]',
            '    [Button: "More Options" id="btn_options_78f9c" actions=[click]]'
        ]
        history = ["filtered pods by 'worker-deployment'"]
        thought = "Target pod 'worker-deployment-78f9c' is located. I need to click 'btn_options_78f9c' to access the pod restart action."
        tool = "click"
        args = {"element_id": "btn_options_78f9c", "delivery_mode": "background"}
        pred = {"expected_element": "menu_action_restart_pod", "timeout_ms": 1000}
        
    elif subcategory == "merge_pr":
        tree = [
            'Window "GitHub - Pull Request #142: refactor(orchestrator): modularize session engine" bounds=(0,0,1920,1080)',
            '  [Badge: "All 18 checks passed (CI/CD, tests, lint)"]',
            '  [Edit: "Add a review comment" id="input_review_comment" value=""]',
            '  [Button: "Approve and Submit Review" id="btn_approve_review" actions=[click]]',
            '  [Button: "Squash and merge" id="btn_squash_merge" actions=[click]]'
        ]
        history = ["reviewed code diff (14 files changed, +340 -120)", "all automated checks green"]
        thought = "The PR is approved and checks are passing. The user specified squash and merge. I will click 'btn_squash_merge'."
        tool = "click"
        args = {"element_id": "btn_squash_merge", "delivery_mode": "background"}
        pred = {"expected_element": "btn_confirm_squash_merge", "timeout_ms": 1500}
        
    elif subcategory == "scale_service":
        tree = [
            'Window "Service Configuration - payment-gateway" bounds=(0,0,1920,1080)',
            '  [Text: "Current Replicas: 3 (Target Max: 20)"]',
            '  [SpinBox: "Desired Instance Count" id="spin_replicas" value="3" min=1 max=50 actions=[set_value,type]]',
            '  [Button: "Apply Scaling Policy" id="btn_apply_scale" actions=[click]]'
        ]
        history = ["opened payment-gateway autoscaling tab"]
        thought = "The user wants to scale replicas from 3 to 10. I will update the SpinBox value to 10."
        tool = "type_text"
        args = {"element_id": "spin_replicas", "text": "10"}
        pred = {"expected_state": {"spin_replicas": "value=10"}, "timeout_ms": 1000}
        
    else: # rotate_secrets
        tree = [
            'Window "Secrets Manager - Database Vault" bounds=(0,0,1920,1080)',
            '  [Text: "Database Secret: db_prod_primary (Version v3 - 45 days old)"]',
            '  [Button: "Rotate Secret Immediately" id="btn_rotate_secret" actions=[click]]',
            '  [Button: "Download Secret Bundle" id="btn_download_bundle" disabled=true]'
        ]
        history = ["authenticated into production vault"]
        thought = "To rotate credentials, I will click 'btn_rotate_secret'. The download button will become enabled once rotation completes."
        tool = "click"
        args = {"element_id": "btn_rotate_secret", "delivery_mode": "background"}
        pred = {"expected_element": "modal_confirm_rotation", "timeout_ms": 2500}

    return goal, tree, history, thought, tool, args, pred

def generate_desktop_trajectory(goal_tuple):
    goal, subcategory = goal_tuple
    
    if subcategory == "file_picker":
        tree = [
            'Window "Open File - Accounting Application" class="CabinetWClass" bounds=(100,100,1024,768)',
            '  [AddressBar: "Address: D:\\Reports\\2026" id="AddressBandRoot"]',
            '  [List: "Items View" id="list_files"]',
            '    [ListItem: "audit_q1.xlsx" id="item_audit_q1"]',
            '    [ListItem: "audit_q2.xlsx" id="item_audit_q2"]',
            '    [ListItem: "audit_q3.xlsx" id="item_audit_q3" actions=[click,double_click]]',
            '    [ListItem: "summary.pdf" id="item_summary"]',
            '  [Edit: "File name:" id="input_file_name" value=""]',
            '  [Button: "Open" id="btn_dialog_open" actions=[click]]'
        ]
        history = ["triggered 'Open Audit File' from application menu"]
        thought = "The target file 'audit_q3.xlsx' is visible in the file list. I will click 'item_audit_q3' and then click Open."
        tool = "click"
        args = {"element_id": "item_audit_q3", "delivery_mode": "background"}
        pred = {"expected_state": {"item_audit_q3": "selected=true"}, "timeout_ms": 800}
        
    elif subcategory == "organon_sprint":
        tree = [
            'Window "Organon Desktop - Orchestrator Cockpit" bounds=(0,0,1920,1080)',
            '  [TabList: "Orchestrator Navigation"]',
            '    [Tab: "Dashboard" active=false]',
            '    [Tab: "Missions" active=false]',
            '    [Tab: "Sessions & Worktrees" id="tab_sessions" active=true]',
            '  [Button: "+ New Sprint Mission" id="btn_new_mission" actions=[click]]',
            '  [Panel: "Active Worktree Instances"]'
        ]
        history = ["switched to Sessions & Worktrees tab"]
        thought = "The user wants to create a new Sprint Mission 'Sprint W45'. Clicking '+ New Sprint Mission' will open the creation modal."
        tool = "click"
        args = {"element_id": "btn_new_mission", "delivery_mode": "background"}
        pred = {"expected_element": "modal_new_mission_form", "timeout_ms": 1200}
        
    elif subcategory == "settings_toggle":
        tree = [
            'Window "System Settings" bounds=(200,150,1200,800)',
            '  [Group: "Personalization Options"]',
            '    [RadioButton: "Light Theme" id="theme_light" checked=true]',
            '    [RadioButton: "Dark Theme" id="theme_dark" checked=false actions=[click]]',
            '    [CheckBox: "Enable High Contrast (4.5:1)" id="check_high_contrast" checked=true]',
            '  [Button: "Save Changes" id="btn_save_settings" actions=[click]]'
        ]
        history = ["opened Settings > Personalization"]
        thought = "User requested toggling Dark Theme. The 'Dark Theme' radio button is currently unchecked. I will click 'theme_dark'."
        tool = "click"
        args = {"element_id": "theme_dark", "delivery_mode": "background"}
        pred = {"expected_state": {"theme_dark": "checked=true"}, "timeout_ms": 800}
        
    else: # print_doc
        tree = [
            'Window "Print" class="#32770" bounds=(400,200,600,500)',
            '  [Combobox: "Select Printer" id="combo_printer" value="Office HP LaserJet"]',
            '  [Edit: "Number of copies:" id="input_copies" value="1" actions=[type]]',
            '  [CheckBox: "Print on both sides" id="check_duplex" checked=false actions=[click]]',
            '  [Button: "Print" id="btn_submit_print" actions=[click]]'
        ]
        history = ["pressed Ctrl+P in document editor"]
        thought = "The print dialog is open. The user requested 2 copies and double-sided. First, set copies to 2 in 'input_copies'."
        tool = "type_text"
        args = {"element_id": "input_copies", "text": "2"}
        pred = {"expected_state": {"input_copies": "value=2"}, "timeout_ms": 500}

    return goal, tree, history, thought, tool, args, pred

def generate_security_edge_cases(goal_tuple):
    goal, subcategory = goal_tuple
    
    if subcategory == "mfa_handshake":
        tree = [
            'Window "Corporate SSO - Two-Factor Authentication" bounds=(0,0,1920,1080)',
            '  [Heading: "Verify Your Identity"]',
            '  [Text: "We sent an SMS security code with 6 digits to your phone ending in **40."]',
            '  [Edit: "Enter 6-digit code" id="input_2fa_code" value="" focused=true]',
            '  [Button: "Verify and Login" id="btn_verify_otp" disabled=true]',
            '  [Link: "Resend security code" id="link_resend"]'
        ]
        history = ["entered corporate email and password", "credentials accepted, 2FA challenge triggered"]
        thought = "A 2FA challenge requires a temporary SMS code delivered to the human operator's phone. As an autonomous agent without access to private SMS streams, I must pause and request operator input."
        tool = "escalate_human"
        args = {
            "reason": "MFA_SMS_REQUIRED",
            "message": "Two-factor SMS security code required to proceed. Please enter the 6-digit code sent to your phone."
        }
        pred = {"expected_state": {"input_2fa_code": "value_len=6"}, "timeout_ms": 60000}
        
    elif subcategory == "captcha_detected":
        tree = [
            'Window "Access Verification - Cloudflare Turnstile" bounds=(0,0,1920,1080)',
            '  [Heading: "Checking if the site connection is secure"]',
            '  [IFrame: "Cloudflare Challenge Box" id="cf_turnstile_box"]',
            '    [CheckBox: "Verify you are human" id="cf_checkbox" checked=false]',
            '  [Text: "Ray ID: 894a73b1892 | IP: 177.18.29.1"]'
        ]
        history = ["requested URL: https://portal.company.com/secure-vault"]
        thought = "A Cloudflare Turnstile anti-bot challenge is actively blocking the page. Autonomous bypass violates safety policies and risks IP banning. Escalating to human operator for resolution."
        tool = "escalate_human"
        args = {
            "reason": "CAPTCHA_CHALLENGE_DETECTED",
            "message": "Cloudflare Turnstile challenge detected. Please complete the verification in the active browser window."
        }
        pred = {"expected_element": "dashboard_root", "timeout_ms": 45000}
        
    elif subcategory == "destructive_action":
        tree = [
            'Window "Account Settings - Danger Zone" bounds=(0,0,1920,1080)',
            '  [Heading: "Permanently Delete Organization Account"]',
            '  [Text: "Warning: This action will permanently wipe all 14 database clusters, 240 active seats and cannot be undone."]',
            '  [Edit: "Type organization name \'Acme-Corp\' to confirm" id="input_confirm_wipe"]',
            '  [Button: "I understand the consequences, delete this organization" id="btn_delete_org_forever" actions=[click]]'
        ]
        history = ["navigated to Organization > Danger Zone"]
        thought = "The requested action is permanently destructive with catastrophic data loss risk. Safe agent policy mandates explicit human confirmation before executing irreversible deletions."
        tool = "escalate_human"
        args = {
            "reason": "DESTRUCTIVE_IRREVERSIBLE_ACTION",
            "message": "High-risk destructive operation: Deleting entire organization and 14 database clusters. Please explicitly authorize this step."
        }
        pred = {"requires_operator_override": True, "timeout_ms": 120000}
        
    else: # scroll_discovery
        tree = [
            'Window "Terms of Service & Licensing" bounds=(0,0,1920,1080)',
            '  [Heading: "Master Services Agreement (Page 1 of 8)"]',
            '  [Text: "1. Scope of License and Permitted Use..."]',
            '  [ScrollBar: "VerticalScroll" position=0 range=[0, 1000]]'
        ]
        history = ["navigated to contract review page"]
        thought = "The target acceptance checkbox 'I accept all terms' is not visible in the current viewport (page 1 of 8). I must scroll down through the document to bring the controls into view."
        tool = "scroll"
        args = {"direction": "down", "by": "page", "amount": 3}
        pred = {"expected_element": "checkbox_accept_terms", "timeout_ms": 2000}

    return goal, tree, history, thought, tool, args, pred

def build_argos_dataset():
    print(f"Generating {TOTAL_SAMPLES} Argos-Action high-precision trajectories...")
    dataset = []
    
    generators = [
        generate_ecommerce_trajectory,
        generate_saas_trajectory,
        generate_devops_trajectory,
        generate_desktop_trajectory,
        generate_security_edge_cases
    ]
    
    for i in range(TOTAL_SAMPLES):
        archetype_idx = i % len(ARCHETYPES)
        archetype = ARCHETYPES[archetype_idx]
        gen_fn = generators[archetype_idx]
        goal_tuple = random.choice(archetype["goals"])
        
        goal, tree_nodes, history, thought, tool, args, pred = gen_fn(goal_tuple)
        
        tree_text = "\n".join(tree_nodes)
        history_text = "\n".join([f"- {h}" for h in history])
        
        user_prompt = (
            f"### User Objective:\n{goal}\n\n"
            f"### Current Screen State (Semantic Accessibility Tree):\n{tree_text}\n\n"
            f"### Execution History:\n{history_text}"
        )
        
        output_payload = {
            "thought": thought,
            "tool_call": {
                "tool": tool,
                "arguments": args
            },
            "verification_predicate": pred
        }
        output_json = json.dumps(output_payload, indent=2, ensure_ascii=False)
        
        sample = {
            "id": f"argos_act_{i+1:05d}",
            "domain": archetype["domain"],
            "instruction": SYSTEM_PROMPT,
            "input": user_prompt,
            "output": output_json,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": json.dumps(output_payload, ensure_ascii=False)}
            ]
        }
        dataset.append(sample)
        
    random.shuffle(dataset)
    split_idx = int(len(dataset) * TRAIN_RATIO)
    train_data = dataset[:split_idx]
    test_data = dataset[split_idx:]
    
    train_file = OUTPUT_DIR / "train.jsonl"
    test_file = OUTPUT_DIR / "test.jsonl"
    
    with open(train_file, "w", encoding="utf-8") as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            
    with open(test_file, "w", encoding="utf-8") as f:
        for item in test_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            
    print(f"Argos-Action dataset generated successfully!")
    print(f"Total: {len(dataset)} | Train: {len(train_data)} | Test: {len(test_data)}")

if __name__ == "__main__":
    build_argos_dataset()
