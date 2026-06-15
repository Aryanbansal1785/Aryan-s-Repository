import json
import random
import time
import uuid
from datetime import datetime, timezone
from dataclasses import dataclass, asdict
from typing import Optional
from kafka import KafkaProducer

# PaySim Transaction Producer

# 3 fraud rules are embedded as patterns:
#   Rule 1: Account Emptied  — balance_orig_after = 0
#   Rule 2: Large Amount     — amount > 200,000
#   Rule 3: Balance Mismatch — abs((before - after) - amount) > 1

KAFKA_BOOTSTRAP = "localhost:29092"
TOPIC = "paysim_transactions"
TRANSACTIONS_PER_SECOND = 10
FRAUD_RATE = 0.15

TRANSACTION_TYPES = ["TRANSFER", "CASH_OUT", "PAYMENT", "DEBIT", "CASH_IN"]
FRAUD_TYPES = ["TRANSFER", "CASH_OUT"]

NORMAL_ACCOUNTS  = [f"C{random.randint(100000000, 999999999)}" for _ in range(200)]
FRAUD_ACCOUNTS   = [f"C{random.randint(100000000, 999999999)}" for _ in range(20)]
DEST_ACCOUNTS    = [f"C{random.randint(100000000, 999999999)}" for _ in range(100)]
MERCHANT_ACCOUNTS = [f"M{random.randint(100000000, 999999999)}" for _ in range(50)]


@dataclass
class PaySimTransaction:
    transaction_id: str
    step_hour: int
    transaction_type: str
    amount: float
    account_origin: str
    balance_orig_before: float
    balance_orig_after: float
    account_dest: str
    balance_dest_before: float
    balance_dest_after: float
    is_fraud: int
    is_flagged_by_system: int
    timestamp: str
    fraud_rule_triggered: Optional[str]


def make_normal_transaction() -> PaySimTransaction:
    tx_type = random.choice(TRANSACTION_TYPES)
    amount = round(random.uniform(1000, 300000), 2)
    balance_before = round(random.uniform(amount, amount * 5), 2)
    balance_after = round(balance_before - amount, 2)
    dest = random.choice(MERCHANT_ACCOUNTS if tx_type in ["PAYMENT", "DEBIT"] else DEST_ACCOUNTS)
    dest_before = round(random.uniform(0, 500000), 2)
    dest_after = round(dest_before + amount, 2)
    return PaySimTransaction(
        transaction_id=str(uuid.uuid4()),
        step_hour=random.randint(1, 743),
        transaction_type=tx_type,
        amount=amount,
        account_origin=random.choice(NORMAL_ACCOUNTS),
        balance_orig_before=balance_before,
        balance_orig_after=balance_after,
        account_dest=dest,
        balance_dest_before=dest_before,
        balance_dest_after=dest_after,
        is_fraud=0,
        is_flagged_by_system=0,
        timestamp=datetime.now(timezone.utc).isoformat(),
        fraud_rule_triggered=None
    )


def make_fraud_rule1_account_emptied() -> PaySimTransaction:
    """
    RULE 1: Account Emptied
    Your SQL: balance_orig_after = 0 AND balance_orig_before > 0
    Criminal drains entire account in one TRANSFER or CASH_OUT.
    """
    tx_type = random.choice(FRAUD_TYPES)
    balance_before = round(random.uniform(50000, 2000000), 2)
    amount = balance_before
    balance_after = 0.0
    dest_before = round(random.uniform(0, 100000), 2)
    dest_after = round(dest_before + amount, 2)
    return PaySimTransaction(
        transaction_id=str(uuid.uuid4()),
        step_hour=random.randint(1, 743),
        transaction_type=tx_type,
        amount=round(amount, 2),
        account_origin=random.choice(FRAUD_ACCOUNTS),
        balance_orig_before=balance_before,
        balance_orig_after=balance_after,
        account_dest=random.choice(DEST_ACCOUNTS),
        balance_dest_before=dest_before,
        balance_dest_after=dest_after,
        is_fraud=1,
        is_flagged_by_system=0,
        timestamp=datetime.now(timezone.utc).isoformat(),
        fraud_rule_triggered="RULE1_ACCOUNT_EMPTIED"
    )


def make_fraud_rule2_large_amount() -> PaySimTransaction:
    """
    RULE 2: Large Amount
    Your SQL: amount > 200000
    Single large transfer above normal transaction size.
    68% of your fraud cases have amount > $200k.
    """
    tx_type = random.choice(FRAUD_TYPES)
    amount = round(random.uniform(200001, 10000000), 2)
    balance_before = round(random.uniform(amount, amount * 1.5), 2)
    balance_after = round(balance_before - amount, 2)
    dest_before = round(random.uniform(0, 500000), 2)
    dest_after = round(dest_before + amount, 2)
    return PaySimTransaction(
        transaction_id=str(uuid.uuid4()),
        step_hour=random.randint(1, 743),
        transaction_type=tx_type,
        amount=amount,
        account_origin=random.choice(FRAUD_ACCOUNTS),
        balance_orig_before=balance_before,
        balance_orig_after=balance_after,
        account_dest=random.choice(DEST_ACCOUNTS),
        balance_dest_before=dest_before,
        balance_dest_after=dest_after,
        is_fraud=1,
        is_flagged_by_system=0,
        timestamp=datetime.now(timezone.utc).isoformat(),
        fraud_rule_triggered="RULE2_LARGE_AMOUNT"
    )


def make_fraud_rule3_balance_mismatch() -> PaySimTransaction:
    """
    RULE 3: Balance Mismatch
    Your SQL: ABS((balance_orig_before - balance_orig_after) - amount) > 1
    Reported amount does not match actual balance change.
    Most technically interesting rule for interviews.
    """
    tx_type = random.choice(FRAUD_TYPES)
    amount = round(random.uniform(50000, 5000000), 2)
    balance_before = round(random.uniform(amount * 1.2, amount * 3), 2)
    actual_deduction = round(amount + random.uniform(100, 50000), 2)
    balance_after = max(0.0, round(balance_before - actual_deduction, 2))
    dest_before = round(random.uniform(0, 200000), 2)
    dest_after = round(dest_before + amount, 2)
    return PaySimTransaction(
        transaction_id=str(uuid.uuid4()),
        step_hour=random.randint(1, 743),
        transaction_type=tx_type,
        amount=amount,
        account_origin=random.choice(FRAUD_ACCOUNTS),
        balance_orig_before=balance_before,
        balance_orig_after=balance_after,
        account_dest=random.choice(DEST_ACCOUNTS),
        balance_dest_before=dest_before,
        balance_dest_after=dest_after,
        is_fraud=1,
        is_flagged_by_system=0,
        timestamp=datetime.now(timezone.utc).isoformat(),
        fraud_rule_triggered="RULE3_BALANCE_MISMATCH"
    )


def create_producer() -> KafkaProducer:
    return KafkaProducer(
        bootstrap_servers=KAFKA_BOOTSTRAP,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        key_serializer=lambda k: k.encode("utf-8"),
        acks="all",
        retries=3,
        linger_ms=10
    )


def run():
    print("=" * 55)
    print("  PaySim Fraud Transaction Producer")
    print("=" * 55)
    print(f"  Kafka:  {KAFKA_BOOTSTRAP}")
    print(f"  Topic:  {TOPIC}")
    print(f"  Rate:   {TRANSACTIONS_PER_SECOND} tx/sec")
    print(f"  Fraud:  {FRAUD_RATE*100:.0f}% of transactions")
    print("=" * 55)
    print("  Rule 1 — Account Emptied  (balance_after = 0)")
    print("  Rule 2 — Large Amount     (amount > $200,000)")
    print("  Rule 3 — Balance Mismatch (amount != balance change)")
    print("=" * 55)
    print("Press Ctrl+C to stop\n")

    producer = create_producer()
    total = 0
    fraud_counts = {"RULE1_ACCOUNT_EMPTIED": 0, "RULE2_LARGE_AMOUNT": 0, "RULE3_BALANCE_MISMATCH": 0}
    start_time = time.time()

    try:
        while True:
            roll = random.random()
            if roll < FRAUD_RATE:
                rule = random.choice(["RULE1_ACCOUNT_EMPTIED", "RULE2_LARGE_AMOUNT", "RULE3_BALANCE_MISMATCH"])
                if rule == "RULE1_ACCOUNT_EMPTIED":
                    tx = make_fraud_rule1_account_emptied()
                elif rule == "RULE2_LARGE_AMOUNT":
                    tx = make_fraud_rule2_large_amount()
                else:
                    tx = make_fraud_rule3_balance_mismatch()
                fraud_counts[rule] += 1
            else:
                tx = make_normal_transaction()

            producer.send(
                topic=TOPIC,
                key=tx.account_origin,
                value=asdict(tx)
            )
            total += 1

            if total % 100 == 0:
                elapsed = time.time() - start_time
                rate = total / elapsed
                print(
                    f"Sent {total:,} tx | {rate:.0f} tx/sec | "
                    f"R1(emptied):{fraud_counts['RULE1_ACCOUNT_EMPTIED']} | "
                    f"R2(large):{fraud_counts['RULE2_LARGE_AMOUNT']} | "
                    f"R3(mismatch):{fraud_counts['RULE3_BALANCE_MISMATCH']}"
                )

            time.sleep(1.0 / TRANSACTIONS_PER_SECOND)

    except KeyboardInterrupt:
        print(f"\nStopped. Total sent: {total:,}")
    finally:
        producer.flush()
        producer.close()
        print("Producer closed.")


if __name__ == "__main__":
    run()