class CreateSummaries < ActiveRecord::Migration
  def self.up
    create_table :summaries do |t|
      # t.column :name, :string
      t.column :id, :primary_key
      t.column :email_thread_id, :integer
      t.column :experiment_id, :integer
      t.column :Sum, :text
      t.column :Sent, :string
      t.column :Label, :string
    end
  end

  def self.down
    drop_table :summaries
  end
end
